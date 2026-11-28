import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.permissions import PermissionPolicy, split_shell_commands


def main() -> None:
    # 1. Compound shell command splitting across &&, ||, ;, | while respecting quotes
    parts = split_shell_commands('echo "a && b" && pytest -q || rm -rf /tmp/bad; ls | grep py')
    assert parts == ['echo "a && b"', "pytest -q", "rm -rf /tmp/bad", "ls", "grep py"], f"Unexpected split: {parts}"

    with tempfile.TemporaryDirectory(prefix="u8_verify_") as tmp:
        ws = Path(tmp) / "proj"
        state = ws / ".broutegpt"
        state.mkdir(parents=True, exist_ok=True)

        settings = {
            "permissions": {
                "allow": [
                    "read_file",
                    "write_file(src/*)",
                    "run_shell(pytest *)",
                    "run_shell(echo *)",
                ],
                "ask": [
                    "run_shell(python3 *)",
                ],
                "deny": [
                    "run_shell(*rm -rf*)",
                    "write_file(secrets/*)",
                ],
            }
        }
        (state / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        # 2. Declarative allow / ask / deny evaluation without auto_guard
        policy = PermissionPolicy(ws, auto_guard=False)

        allow_res = policy.evaluate({"action": "write_file", "path": "src/main.py"})
        assert allow_res.allowed and allow_res.decision == "allow"

        deny_file = policy.evaluate({"action": "write_file", "path": "secrets/key.txt"})
        assert not deny_file.allowed and deny_file.decision == "deny"

        # Compound command where first segment is allowed, second segment matches deny
        compound_deny = policy.evaluate({"action": "run_shell", "command": "echo hi && rm -rf /tmp/test"})
        assert not compound_deny.allowed and compound_deny.decision == "deny"

        # 3. Auto-Guard fast free-model classifier on 'ask' commands
        mock_router = MagicMock()
        mock_router.query_autonomous.side_effect = [
            ('{"safe": true, "reason": "Runs local unit test script"}', "groq:llama-3.1-8b-instant"),
            ('{"safe": false, "reason": "Exfiltrates environment secrets"}', "groq:llama-3.1-8b-instant"),
        ]
        guard_policy = PermissionPolicy(ws, router=mock_router, auto_guard=True)

        safe_ask = guard_policy.evaluate({"action": "run_shell", "command": "python3 verify_dp1.py"})
        assert safe_ask.allowed and safe_ask.decision == "allow"
        assert "llama-3.1-8b-instant" in safe_ask.guard_verdict

        unsafe_ask = guard_policy.evaluate({"action": "run_shell", "command": "python3 -c 'import os; print(os.environ)'"})
        assert not unsafe_ask.allowed and unsafe_ask.decision == "deny"

        # 4. End-to-end DPOrchestrator enforcement & --auto-guard CLI flag
        mock_agent_router = MagicMock()
        mock_agent_router.router = MagicMock()
        mock_agent_router.hook_engine = None
        mock_agent_router.call_log = []
        mock_agent_router.get_active_input_limit.return_value = 32000

        plan_json = json.dumps({
            "project_name": "perm_demo",
            "overview": "Permission policy test",
            "entrypoint": "src/app.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Test permissions",
                    "description": "Write verify_dp1.py",
                    "deliverables": ["verify_dp1.py"],
                    "verification_criteria": "verify_dp1.py exits 0",
                }
            ],
        })
        turns = [
            plan_json,
            # Denied by rule write_file(secrets/*)
            json.dumps({"thought": "try forbidden path", "action": "write_file", "path": "secrets/token.txt", "content": "x"}),
            # Allowed write + complete
            json.dumps({
                "thought": "write verify_dp1 and complete",
                "actions": [
                    {"action": "write_file", "path": "verify_dp1.py", "content": "assert True\n"},
                    {"action": "dp_complete", "summary": "done"},
                ],
            }),
        ]
        turn_iter = iter(turns)
        mock_agent_router.query_autonomous.side_effect = lambda *a, **kw: (next(turn_iter), "gemini:m-ok")

        orch = DPOrchestrator(ws, router=mock_agent_router, auto_main_git=False, auto_guard=True)
        assert orch.permissions.auto_guard is True
        orch.run("Verify permission enforcement")

        assert not (ws / "secrets" / "token.txt").exists(), "Denied file write should never touch disk"
        all_events = "\n".join(e.content for e in orch.stream.events)
        assert "PERMISSION DENIED (deny)" in all_events

        help_proc = subprocess.run(
            [sys.executable, str(ROOT / "broutegpt.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--auto-guard" in help_proc.stdout

    print("VERIFY_U8_PERMISSIONS: PASS")


if __name__ == "__main__":
    main()

