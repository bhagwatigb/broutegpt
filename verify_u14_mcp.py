import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["BROUTEGPT_NO_SANDBOX"] = "1"

from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.mcp_bridge import MCPRegistry


MOCK_MCP_SERVER_CODE = '''import json
import sys

TOOLS = [
    {
        "name": "compute_checksum",
        "description": "Compute a deterministic demo checksum for input text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text payload to hash"},
                "salt": {"type": "string", "description": "Optional salt suffix"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "unused_heavy_tool",
        "description": "A bulky tool whose schema should stay deferred until explicitly loaded.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "VERY_VERBOSE_SCHEMA_SENTINEL": {"type": "string"},
            },
        },
    },
]

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    req = json.loads(line)
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}
    if req_id is None and method.startswith("notifications/"):
        continue
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-demo-mcp", "version": "0.1.0"},
            },
        }
    elif method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        t_name = params.get("name")
        t_args = params.get("arguments") or {}
        if t_name == "compute_checksum":
            payload = f"{t_args.get('text', '')}:{t_args.get('salt', 'default')}"
            total = sum(ord(ch) for ch in payload)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"CHK_{total}_{payload}"}],
                    "isError": False,
                },
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool {t_name}"}],
                    "isError": True,
                },
            }
    else:
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method {method}"},
        }
    sys.stdout.write(json.dumps(resp) + "\\n")
    sys.stdout.flush()
'''


class MockMCPRouter:
    """Drives DP 1 through load_mcp_tool -> call_mcp_tool -> write_file -> dp_complete."""

    def __init__(self) -> None:
        self.router = None
        self.hook_engine = None
        self.call_log = []
        self.turn = 0
        self.prompts_seen = []

    def query_autonomous(self, prompt: str, **kwargs):
        self.turn += 1
        self.prompts_seen.append(prompt)
        history = kwargs.get("history") or []
        hist_text = "\n".join(str(m.get("content", "")) for m in history)

        if self.turn == 1:
            reply = json.dumps({
                "thought": "Lazily loading schema for demo__compute_checksum before calling it.",
                "action": "load_mcp_tool",
                "name": "demo__compute_checksum",
            })
        elif self.turn == 2:
            assert "inputSchema" in hist_text and "salt" in hist_text, f"Expected hydrated schema in history: {hist_text}"
            reply = json.dumps({
                "thought": "Calling demo__compute_checksum with text and salt.",
                "action": "call_mcp_tool",
                "name": "demo__compute_checksum",
                "arguments": {"text": "broutegpt", "salt": "v14"},
            })
        else:
            assert "CHK_" in hist_text, f"Expected MCP tool result in history: {hist_text}"
            val = [ln.strip() for ln in hist_text.splitlines() if "CHK_" in ln][-1]
            reply = json.dumps({
                "thought": "Writing verified checksum output and verification script.",
                "actions": [
                    {
                        "action": "write_file",
                        "path": "mcp_result.py",
                        "content": f"RESULT = {val!r}\n",
                    },
                    {
                        "action": "write_file",
                        "path": "verify_dp1.py",
                        "content": "import mcp_result\nassert 'broutegpt:v14' in mcp_result.RESULT\nprint('DP1_MCP_OK')\n",
                    },
                    {"action": "run_shell", "command": "python3 verify_dp1.py"},
                    {"action": "dp_complete", "summary": "Executed lazy MCP tool and verified output"},
                ],
            })

        self.call_log.append({"status": "ok", "prompt_tokens": 100, "completion_tokens": 60})
        return reply, "mock-mcp-model"


def test_lazy_mcp_registry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mock_srv = root / "mock_mcp_server.py"
        mock_srv.write_text(MOCK_MCP_SERVER_CODE, encoding="utf-8")

        state_dir = root / ".broutegpt"
        state_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "mcpServers": {
                "demo": {
                    "command": sys.executable,
                    "args": [str(mock_srv)],
                }
            }
        }
        (state_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        reg = MCPRegistry(root)
        try:
            prompt_idx = reg.get_prompt_index()
            assert "demo__compute_checksum" in prompt_idx
            assert "demo__unused_heavy_tool" in prompt_idx
            assert "VERY_VERBOSE_SCHEMA_SENTINEL" not in prompt_idx, "Lazy prompt index leaked full schema!"

            loaded = reg.load_mcp_tool("demo__compute_checksum")
            assert loaded["found"] is True
            assert "salt" in loaded["inputSchema"]["properties"]

            out = reg.call_mcp_tool("demo__compute_checksum", {"text": "hello", "salt": "42"})
            assert out.startswith("CHK_") and "hello:42" in out
        finally:
            reg.close_all()
    print("[PASS] 1. MCPRegistry lazy tool discovery, deferred schema hydration, and tools/call")


def test_orchestrator_mcp_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mock_srv = root / "mock_mcp_server.py"
        mock_srv.write_text(MOCK_MCP_SERVER_CODE, encoding="utf-8")

        state_dir = root / ".broutegpt"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "settings.json").write_text(
            json.dumps({
                "mcpServers": {
                    "demo": {
                        "command": sys.executable,
                        "args": [str(mock_srv)],
                    }
                }
            }, indent=2),
            encoding="utf-8",
        )
        plan = {
            "project_name": "mcp_orch_demo",
            "overview": "Use lazy MCP tool during DP 1.",
            "entrypoint": "mcp_result.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Run MCP Checksum",
                    "depends_on": [],
                    "description": "Load and invoke demo__compute_checksum",
                    "deliverables": ["mcp_result.py", "verify_dp1.py"],
                    "verification_criteria": "mcp_result.RESULT contains broutegpt:v14",
                    "status": "pending",
                }
            ],
        }
        (state_dir / "dp_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        mock_router = MockMCPRouter()
        orch = DPOrchestrator(root, router=mock_router, auto_main_git=False)
        report = orch.run(goal="Test lazy MCP inside orchestrator", resume=True)

        assert report["total_dps"] == 1
        assert "demo__compute_checksum" in mock_router.prompts_seen[0]
        assert "VERY_VERBOSE_SCHEMA_SENTINEL" not in mock_router.prompts_seen[0]
        assert (root / "mcp_result.py").is_file()
    print("[PASS] 2. DPOrchestrator lazy load_mcp_tool + call_mcp_tool end-to-end execution")


def test_broutegpt_mcp_server_mode() -> None:
    repo_root = Path(__file__).resolve().parent
    broutegpt_py = repo_root / "broutegpt.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "sample_mod.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

        proc = subprocess.Popen(
            [sys.executable, "-u", str(broutegpt_py), "--mcp", "--project-dir", str(tmp_path)],
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None

            def rpc(payload):
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                if payload.get("id") is None:
                    return None
                line = proc.stdout.readline()
                assert line, f"No stdout reply from --mcp server. STDERR: {proc.stderr.read() if proc.stderr else ''}"
                return json.loads(line)

            init_resp = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert init_resp["result"]["serverInfo"]["name"] == "broutegpt"

            rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

            list_resp = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            tool_names = [t["name"] for t in list_resp["result"]["tools"]]
            for expected in ("broutegpt_doctor", "broutegpt_repo_map", "broutegpt_plan", "broutegpt_chat"):
                assert expected in tool_names, f"Missing MCP server tool {expected}: {tool_names}"

            doc_resp = rpc({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "broutegpt_doctor", "arguments": {}},
            })
            assert doc_resp["result"]["isError"] is False
            doc_json = json.loads(doc_resp["result"]["content"][0]["text"])
            assert doc_json["healthy"] is True

            map_resp = rpc({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "broutegpt_repo_map", "arguments": {"project_dir": str(tmp_path)}},
            })
            assert map_resp["result"]["isError"] is False
            assert "sample_mod.py" in map_resp["result"]["content"][0]["text"]
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
    print("[PASS] 3. broutegpt --mcp stdio JSON-RPC server (initialize, tools/list, tools/call)")


if __name__ == "__main__":
    test_lazy_mcp_registry()
    test_orchestrator_mcp_integration()
    test_broutegpt_mcp_server_mode()
    print("\nALL DP 14 MCP CLIENT, LAZY TOOL SEARCH & --MCP SERVER CHECKS PASSED!")

