import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.condenser import RollingCondenser
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.event_stream import EventStream
from broutegpt_agent.hook_engine import HookEngine, LIFECYCLE_EVENTS
from broutegpt_agent.router_adapter import CloudFirstRouter


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="u7_verify_") as tmp:
        ws = Path(tmp) / "proj"
        state = ws / ".broutegpt"
        state.mkdir(parents=True, exist_ok=True)
        log_file = ws / "hook_trace.log"

        hooks_cfg = {
            "hooks": {
                "SessionStart": [
                    {"command": f'echo "SESSION_CTX" && echo "SessionStart" >> "{log_file}"'}
                ],
                "PreToolUse": [
                    {
                        "matcher": "run_shell",
                        "command": 'python3 -c "import os,sys; p=os.environ.get(\'BROUTEGPT_HOOK_PAYLOAD\',\'\'); sys.exit(2) if \'forbidden_cmd\' in p else print(\'PRE_TOOL_OK\')"',
                    },
                    {"command": f'echo "PreToolUse:$BROUTEGPT_HOOK_TOOL" >> "{log_file}"'},
                ],
                "PostToolUse": [
                    {"command": f'echo "POST_CTX" && echo "PostToolUse:$BROUTEGPT_HOOK_TOOL" >> "{log_file}"'}
                ],
                "PostToolBatch": [
                    {"command": f'echo "BATCH_CTX" && echo "PostToolBatch" >> "{log_file}"'}
                ],
                "PreCompact": [
                    {"command": f'echo "COMPACT_CTX" && echo "PreCompact" >> "{log_file}"'}
                ],
                "PreModelSwitch": [
                    {"command": f'echo "SWITCH_CTX" && echo "PreModelSwitch" >> "{log_file}"'}
                ],
                "DPVerifyFailed": [
                    {"command": f'echo "VERIFY_FAIL_CTX" && echo "DPVerifyFailed" >> "{log_file}"'}
                ],
                "Stop": [
                    {"command": f'echo "STOP_CTX" && echo "Stop" >> "{log_file}"'}
                ],
            }
        }
        (state / "hooks.json").write_text(json.dumps(hooks_cfg, indent=2), encoding="utf-8")

        engine = HookEngine(ws)
        assert len(LIFECYCLE_EVENTS) == 8

        # 1. Verify PreToolUse exit 2 blocks command & stdout injects additionalContext
        allowed_res = engine.dispatch("PreToolUse", {"tool": "run_shell", "input": {"command": "echo hi"}})
        assert not allowed_res.blocked
        assert "PRE_TOOL_OK" in allowed_res.additional_context

        blocked_res = engine.dispatch("PreToolUse", {"tool": "run_shell", "input": {"command": "forbidden_cmd"}})
        assert blocked_res.blocked is True

        # 2. Verify PreCompact hook in RollingCondenser
        stream = EventStream(ws)
        stream.append("goal", "Test hooks", dp_index=0)
        for i in range(8):
            stream.append("action", f'{{"action": "list_dir", "path": "."}}', dp_index=1)
            stream.append("observation", f"obs {i}", dp_index=1)
        cond = RollingCondenser(state, keep_tail_events=2, condense_batch_min=2, hook_engine=engine)
        hist_res = cond.compile_history(stream, target_token_budget=8000, active_dp_index=1)
        joined_hist = "\n".join(m["content"] for m in hist_res["history"])
        assert "COMPACT_CTX" in joined_hist, "PreCompact additionalContext missing from condensed history"

        # 3. Verify PreModelSwitch hook in CloudFirstRouter
        mock_base = MagicMock()
        mock_base.models_by_option = {
            "1": [],
            "2": [
                {"id": "m-fail", "provider": "gemini", "input_limit": 32000, "output_limit": 4096, "chat": True},
                {"id": "m-ok", "provider": "gemini", "input_limit": 32000, "output_limit": 4096, "chat": True},
            ],
        }
        mock_base.role_config = {}
        mock_base._cooldown_until = {}
        mock_base.last_call_usage = {}
        seen_prompts = []

        def fake_call(prov, mid, prompt, *a, **kw):
            seen_prompts.append((mid, prompt))
            if mid == "m-fail":
                raise RuntimeError("429 rate limit")
            return '{"thought":"ok","action":"list_dir","path":"."}'

        mock_base._call_model.side_effect = fake_call
        cfr = CloudFirstRouter(router=mock_base, hook_engine=engine)
        out_txt, used_mod = cfr.query_autonomous("Initial prompt", tier="speed")
        assert used_mod == "gemini:m-ok"
        assert any(mid == "m-ok" and "SWITCH_CTX" in pr for mid, pr in seen_prompts), "PreModelSwitch additionalContext missing"

        # 4. Verify end-to-end DPOrchestrator lifecycle hooks (SessionStart, PreToolUse block & allow, PostToolUse, PostToolBatch, DPVerifyFailed, Stop)
        mock_agent_router = MagicMock()
        mock_agent_router.router = mock_base
        mock_agent_router.hook_engine = None
        mock_agent_router.call_log = []
        mock_agent_router.get_active_input_limit.return_value = 32000

        plan_json = json.dumps({
            "project_name": "hook_demo",
            "overview": "Hook test",
            "entrypoint": "app.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Build and verify",
                    "description": "Write verify_dp1.py",
                    "deliverables": ["verify_dp1.py"],
                    "verification_criteria": "verify_dp1.py exits 0",
                }
            ],
        })
        turns = [
            plan_json,
            # Turn 1: blocked PreToolUse on forbidden_cmd
            json.dumps({"thought": "try blocked", "action": "run_shell", "command": "forbidden_cmd"}),
            # Turn 2: call dp_complete BEFORE verify_dp1.py exists -> triggers DPVerifyFailed
            json.dumps({"thought": "premature complete", "action": "dp_complete", "summary": "done"}),
            # Turn 3: multi-action batch creating verify_dp1.py and completing DP 1 -> triggers PostToolBatch & Stop
            json.dumps({
                "thought": "write verify_dp1 and finish",
                "actions": [
                    {"action": "write_file", "path": "verify_dp1.py", "content": "assert 1 + 1 == 2\nprint('OK')\n"},
                    {"action": "dp_complete", "summary": "verified"},
                ],
            }),
        ]
        turn_iter = iter(turns)
        mock_agent_router.query_autonomous.side_effect = lambda *a, **kw: (next(turn_iter), "gemini:m-ok")

        orch = DPOrchestrator(ws, router=mock_agent_router, auto_main_git=False)
        report = orch.run("Test all lifecycle hooks")
        assert report["stop_hook_context"] == "STOP_CTX"

        all_obs = "\n".join(e.content for e in orch.stream.events)
        assert "SESSION_CTX" in all_obs
        assert "BLOCKED BY PreToolUse HOOK" in all_obs
        assert "POST_CTX" in all_obs
        assert "BATCH_CTX" in all_obs
        assert "VERIFY_FAIL_CTX" in all_obs
        assert "STOP_CTX" in all_obs

        trace_lines = log_file.read_text(encoding="utf-8").splitlines()
        for expected_evt in ("SessionStart", "PreToolUse", "PostToolUse", "PostToolBatch", "PreCompact", "PreModelSwitch", "DPVerifyFailed", "Stop"):
            assert any(ln.startswith(expected_evt) for ln in trace_lines), f"Missing trace for {expected_evt}"

    print("VERIFY_U7_HOOKS: PASS")


if __name__ == "__main__":
    main()

