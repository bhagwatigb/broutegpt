import contextlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPToolStub:
    """Lightweight tool index entry kept in prompts until full schema is lazily hydrated."""

    server: str
    name: str
    qualified_name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    schema_loaded: bool = False


class StdioMCPClient:
    """Zero-dependency JSON-RPC 2.0 stdio client for a single configured MCP server."""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        timeout: float = 30.0,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen[str]] = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        child_env = os.environ.copy()
        child_env.update({str(k): str(v) for k, v in self.env.items()})
        self._proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(self.cwd) if self.cwd else None,
            env=child_env,
        )
        self._initialized = False

    def _send_message(self, payload: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _read_response(self, expected_id: int) -> Dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw_line = self._proc.stdout.readline()
            if not raw_line:
                err_txt = ""
                if self._proc.stderr is not None:
                    try:
                        err_txt = self._proc.stderr.read().strip()
                    except Exception:
                        err_txt = ""
                raise RuntimeError(
                    f"MCP server '{self.server_name}' closed stdout unexpectedly"
                    + (f": {err_txt}" if err_txt else ".")
                )

            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.lower().startswith("content-length:"):
                try:
                    length = int(stripped.split(":", 1)[1].strip())
                except Exception:
                    continue
                while True:
                    hdr = self._proc.stdout.readline()
                    if not hdr or hdr.strip() == "":
                        break
                body = self._proc.stdout.read(length)
                msg = json.loads(body)
            else:
                try:
                    msg = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

            if not isinstance(msg, dict):
                continue
            if msg.get("id") != expected_id:
                continue
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                raise RuntimeError(f"MCP error from '{self.server_name}': {err}")
            return msg.get("result") or {}

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            self._ensure_started()
            if not self._initialized and method != "initialize":
                self._handshake_locked()
            self._req_id += 1
            req_id = self._req_id
            req: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
            }
            if params is not None:
                req["params"] = params
            self._send_message(req)
            return self._read_response(req_id)

    def _handshake_locked(self) -> None:
        self._req_id += 1
        req_id = self._req_id
        init_req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "broutegpt", "version": "1.0.0"},
            },
        }
        self._send_message(init_req)
        self._read_response(req_id)
        self._send_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools")
        return tools if isinstance(tools, list) else []

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        result = self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": dict(arguments or {}),
            },
        )
        is_error = bool(result.get("isError", False))
        blocks = result.get("content")
        rendered_parts: List[str] = []
        if isinstance(blocks, list):
            for blk in blocks:
                if isinstance(blk, dict):
                    if "text" in blk:
                        rendered_parts.append(str(blk["text"]))
                    else:
                        rendered_parts.append(json.dumps(blk, ensure_ascii=False))
                else:
                    rendered_parts.append(str(blk))
        elif result:
            rendered_parts.append(json.dumps(result, ensure_ascii=False))

        output = "\n".join(p for p in rendered_parts if p).strip() or "(empty MCP tool output)"
        if is_error:
            raise RuntimeError(output)
        return output

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        self._initialized = False


class MCPRegistry:
    """
    Lazy MCP client registry backed by `.broutegpt/settings.json` (`mcpServers`).
    Keeps full JSON schemas out of the LLM prompt until `load_mcp_tool(name)` is called.
    """

    def __init__(
        self,
        project_dir: str | Path,
        enabled: bool = True,
        server_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.enabled = bool(enabled)
        self._server_configs: Dict[str, Dict[str, Any]] = {}
        self._clients: Dict[str, StdioMCPClient] = {}
        self._catalog: Dict[str, MCPToolStub] = {}
        self._discovered = False

        if self.enabled:
            if server_configs is not None:
                self._server_configs = dict(server_configs)
            else:
                self._server_configs = self._load_server_configs()

    def _load_server_configs(self) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        candidates = [
            Path.home() / ".broutegpt" / "settings.json",
            self.project_root / ".broutegpt" / "settings.json",
        ]
        for cfg_path in candidates:
            if not cfg_path.is_file():
                continue
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                servers = raw.get("mcpServers") if isinstance(raw, dict) else None
                if isinstance(servers, dict):
                    for s_name, s_cfg in servers.items():
                        if isinstance(s_cfg, dict) and s_cfg.get("command"):
                            merged[str(s_name)] = s_cfg
            except Exception:
                continue
        return merged

    @property
    def configured_servers(self) -> List[str]:
        return sorted(self._server_configs.keys())

    def _get_client(self, server_name: str) -> StdioMCPClient:
        if server_name not in self._clients:
            cfg = self._server_configs[server_name]
            self._clients[server_name] = StdioMCPClient(
                server_name=server_name,
                command=str(cfg["command"]),
                args=[str(a) for a in (cfg.get("args") or [])],
                env=cfg.get("env") if isinstance(cfg.get("env"), dict) else None,
                cwd=self.project_root,
            )
        return self._clients[server_name]

    def discover_tools(self, force: bool = False) -> List[MCPToolStub]:
        if not self.enabled or not self._server_configs:
            return []
        if self._discovered and not force:
            return list(self._catalog.values())

        for s_name in self.configured_servers:
            try:
                client = self._get_client(s_name)
                raw_tools = client.list_tools()
                for t in raw_tools:
                    if not isinstance(t, dict) or not t.get("name"):
                        continue
                    t_name = str(t["name"])
                    q_name = f"{s_name}__{t_name}"
                    existing = self._catalog.get(q_name)
                    self._catalog[q_name] = MCPToolStub(
                        server=s_name,
                        name=t_name,
                        qualified_name=q_name,
                        description=str(t.get("description") or "").strip(),
                        input_schema=dict(t.get("inputSchema") or {}),
                        schema_loaded=existing.schema_loaded if existing else False,
                    )
            except Exception:
                continue

        self._discovered = True
        return list(self._catalog.values())

    def get_prompt_index(self) -> str:
        """Compact, zero-schema prompt summary listing available MCP tools for lazy loading."""
        stubs = self.discover_tools()
        if not stubs:
            return ""
        lines = [
            "AVAILABLE MCP TOOLS (schemas deferred — call `load_mcp_tool` before `call_mcp_tool` to inspect parameters):"
        ]
        for st in sorted(stubs, key=lambda x: x.qualified_name):
            short_desc = (st.description or "No description").splitlines()[0][:100]
            state_tag = "schema loaded" if st.schema_loaded else "lazy"
            lines.append(f"  - {st.qualified_name} [{state_tag}]: {short_desc}")
        return "\n".join(lines)

    def _resolve_stub(self, name: str) -> Optional[MCPToolStub]:
        self.discover_tools()
        key = name.strip()
        if key in self._catalog:
            return self._catalog[key]
        matches = [st for st in self._catalog.values() if st.name == key]
        if len(matches) == 1:
            return matches[0]
        return None

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        self.discover_tools()
        q = query.strip().lower()
        hits: List[Dict[str, Any]] = []
        for st in sorted(self._catalog.values(), key=lambda x: x.qualified_name):
            hay = f"{st.qualified_name} {st.name} {st.description}".lower()
            if not q or q in hay:
                hits.append({
                    "name": st.qualified_name,
                    "server": st.server,
                    "description": st.description,
                    "schema_loaded": st.schema_loaded,
                })
        return hits

    def load_mcp_tool(self, name: str = "", query: str = "") -> Dict[str, Any]:
        """Hydrates the full JSON inputSchema for one MCP tool (or searches matching stubs)."""
        target_name = name.strip()
        if target_name:
            stub = self._resolve_stub(target_name)
            if stub is not None:
                stub.schema_loaded = True
                return {
                    "found": True,
                    "name": stub.qualified_name,
                    "server": stub.server,
                    "tool": stub.name,
                    "description": stub.description,
                    "inputSchema": stub.input_schema,
                }
            # Fall back to fuzzy search if exact name did not match
            search_q = query.strip() or target_name
            return {
                "found": False,
                "query": search_q,
                "matches": self.search_tools(search_q),
            }

        return {
            "found": False,
            "query": query.strip(),
            "matches": self.search_tools(query),
        }

    def call_mcp_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        stub = self._resolve_stub(name)
        if stub is None:
            raise ValueError(
                f"MCP tool '{name}' not found. Available: {sorted(self._catalog.keys())}"
            )
        client = self._get_client(stub.server)
        return client.call_tool(stub.name, arguments=arguments or {})

    def close_all(self) -> None:
        for client in list(self._clients.values()):
            client.close()
        self._clients.clear()


# ============================================================================
# `broutegpt --mcp` STDIO JSON-RPC 2.0 SERVER MODE
# ============================================================================

BROUTEGPT_MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "broutegpt_doctor",
        "description": "Inspect broutegpt environment readiness (Python, Git, OS sandbox, providers, Ollama, model cache).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "env_path": {"type": "string", "description": "Optional path to .env file"},
            },
        },
    },
    {
        "name": "broutegpt_repo_map",
        "description": "Generate a compact codebase map for a workspace directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Workspace directory path"},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (default 6000)"},
            },
        },
    },
    {
        "name": "broutegpt_plan",
        "description": "Generate a read-only Decision-Point (DP) execution plan JSON without modifying workspace source files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Project build specification"},
                "project_dir": {"type": "string", "description": "Target workspace directory"},
                "max_dps": {"type": "integer", "description": "Maximum Decision Points (default 6)"},
                "bare": {"type": "boolean", "description": "Skip AGENTS.md and MEMORY.md rules"},
                "option": {"type": "string", "description": "Optional FreeLLMRouter option (e.g. '1', '8', 'deep')"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "broutegpt_chat",
        "description": "Send a one-shot prompt to broutegpt's multi-provider FreeLLMRouter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt to execute"},
                "system_prompt": {"type": "string", "description": "Optional system prompt"},
                "option": {"type": "string", "description": "Optional router option (1-10 or provider:model)"},
            },
            "required": ["prompt"],
        },
    },
]


def _dispatch_server_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    default_project_dir: Path,
) -> str:
    if tool_name == "broutegpt_doctor":
        import broutegpt

        env_path = str(arguments.get("env_path") or ".env")
        with contextlib.redirect_stdout(sys.stderr):
            report = broutegpt.run_doctor(env_path=env_path, output_format="text")
        return json.dumps(report, indent=2, ensure_ascii=False)

    if tool_name == "broutegpt_repo_map":
        from .repo_map import RepoMapper
        from .terminal_executor import TerminalExecutor

        target_dir = Path(str(arguments.get("project_dir") or default_project_dir)).expanduser().resolve()
        max_chars = int(arguments.get("max_chars") or 6000)
        with contextlib.redirect_stdout(sys.stderr):
            executor = TerminalExecutor(target_dir)
            mapper = RepoMapper(executor)
            return mapper.build_map(max_chars=max_chars)

    if tool_name == "broutegpt_plan":
        import broutegpt
        from .dp_orchestrator import DPOrchestrator
        from .router_adapter import CloudFirstRouter

        goal = str(arguments.get("goal") or "").strip()
        if not goal:
            raise ValueError("broutegpt_plan requires a non-empty 'goal' argument.")
        target_dir = Path(str(arguments.get("project_dir") or default_project_dir)).expanduser().resolve()
        max_dps = int(arguments.get("max_dps") or 6)
        bare = bool(arguments.get("bare", False))
        opt = str(arguments.get("option") or "deep")

        with contextlib.redirect_stdout(sys.stderr):
            base_router = broutegpt.FreeLLMRouter()
            base_router.configure_non_interactive(opt)
            c_router = CloudFirstRouter(router=base_router)
            orch = DPOrchestrator(
                project_dir=target_dir,
                router=c_router,
                max_dps=max_dps,
                bare=bare,
            )
            plan = orch.plan_only(goal=goal)
        return json.dumps(plan, indent=2, ensure_ascii=False)

    if tool_name == "broutegpt_chat":
        import broutegpt

        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("broutegpt_chat requires a non-empty 'prompt' argument.")
        sys_prompt = arguments.get("system_prompt")
        opt = str(arguments.get("option") or "8")

        with contextlib.redirect_stdout(sys.stderr):
            base_router = broutegpt.FreeLLMRouter()
            base_router.configure_non_interactive(opt)
            text, used_model = base_router.query(
                prompt,
                system_prompt=str(sys_prompt) if sys_prompt else None,
            )
        return json.dumps({"model": used_model, "response": text}, indent=2, ensure_ascii=False)

    raise ValueError(f"Unknown broutegpt MCP tool: {tool_name}")


def handle_mcp_jsonrpc_request(
    req: Dict[str, Any],
    default_project_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Pure request handler for unit tests and the stdio MCP server loop."""
    req_id = req.get("id")
    method = str(req.get("method") or "")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    # Notifications have no id and expect no reply
    if req_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "broutegpt", "version": "1.0.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": BROUTEGPT_MCP_TOOLS},
        }

    if method == "tools/call":
        t_name = str(params.get("name") or "")
        t_args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            text_out = _dispatch_server_tool(t_name, t_args, default_project_dir)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text_out}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_mcp_server(project_dir: str | Path = ".") -> None:
    """Runs broutegpt as a stdio JSON-RPC 2.0 MCP server."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    root = Path(project_dir).expanduser().resolve()
    while True:
        raw_line = sys.stdin.readline()
        if not raw_line:
            break
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.lower().startswith("content-length:"):
            try:
                length = int(stripped.split(":", 1)[1].strip())
            except Exception:
                continue
            while True:
                hdr = sys.stdin.readline()
                if not hdr or hdr.strip() == "":
                    break
            body = sys.stdin.read(length)
            try:
                req = json.loads(body)
            except Exception:
                continue
        else:
            try:
                req = json.loads(stripped)
            except Exception:
                continue

        if not isinstance(req, dict):
            continue

        resp = handle_mcp_jsonrpc_request(req, default_project_dir=root)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()

