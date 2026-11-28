import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.checkpoint_manager import CheckpointManager
from broutegpt_agent.cli_runner import run_agent_cli
from broutegpt_agent.code_editor import validate_action_json
from broutegpt_agent.condenser import RollingCondenser
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.event_stream import EventStream
from broutegpt_agent.tool_schemas import AGENT_TOOL_SPECS


def main() -> None:
    # 1. Validate consult_advisor schema in both JSON validator and tool_schemas
    assert any(t["name"] == "consult_advisor" for t in AGENT_TOOL_SPECS)
    assert validate_action_json('{"thought":"need help","action":"consult_advisor","question":"How to parse edge case?"}') is None
    assert validate_action_json('{"thought":"missing q","action":"consult_advisor"}') is not None

    with tempfile.TemporaryDirectory(prefix="u10_verify_") as tmp:
        ws = Path(tmp) / "proj"
        ws.mkdir(parents=True, exist_ok=True)

        # 2. Test on-demand consult_advisor inside DPOrchestrator during a speed-tier step
        mock_router = MagicMock()
        mock_router.router = MagicMock()
        mock_router.hook_engine = None
        mock_router.call_log = []
        mock_router.get_active_input_limit.return_value = 32000

        seen_tiers = []
        plan_json = json.dumps({
            "project_name": "advisor_demo",
            "overview": "Advisor and branching test",
            "entrypoint": "app.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Build module",
                    "description": "Write app.py and verify_dp1.py",
                    "deliverables": ["app.py", "verify_dp1.py"],
                    "verification_criteria": "verify_dp1.py exits 0",
                }
            ],
        })
        turns = [
            plan_json,
            # Step 1 (deep): write initial app.py
            json.dumps({"thought": "write initial app", "action": "write_file", "path": "app.py", "content": "VAL = 1\n"}),
            # Step 2 (speed): consult deep advisor on demand
            json.dumps({"thought": "consult deep advisor", "action": "consult_advisor", "question": "Should VAL be 2?"}),
            # Step 3 (speed): write verify_dp1.py and complete DP 1
            json.dumps({
                "thought": "finish dp1",
                "actions": [
                    {"action": "write_file", "path": "verify_dp1.py", "content": "import app\nassert app.VAL == 1\n"},
                    {"action": "dp_complete", "summary": "verified"},
                ],
            }),
        ]
        turn_iter = iter(turns)

        def fake_query(*args, **kwargs):
            tier = kwargs.get("tier", "smart")
            seen_tiers.append(tier)
            if "QUESTION FROM FAST WORKER" in kwargs.get("prompt", ""):
                return ("Yes, keep VAL = 1 for baseline compatibility.", "gemini:gemini-2.5-pro")
            return (next(turn_iter), f"mock:{tier}")

        mock_router.query_autonomous.side_effect = fake_query

        orch = DPOrchestrator(ws, router=mock_router, auto_main_git=False)
        orch.run("Verify consult_advisor and decoupled rewind/branching")

        assert "deep" in seen_tiers and "speed" in seen_tiers
        obs_all = "\n".join(e.content for e in orch.stream.events)
        assert "ADVISOR RESPONSE (gemini:gemini-2.5-pro)" in obs_all

        # 3. Test --branch <step> session forking
        ckpt = CheckpointManager(ws, auto_main_git=False)
        step_after_app = next(r.step_id for r in ckpt.records if r.label == "write:app.py")
        br_info = run_agent_cli(
            project_dir=str(ws),
            branch_step_id=step_after_app,
            branch_name="exp-branch",
        )
        branch_path = Path(br_info["branch_dir"])
        assert (branch_path / "app.py").exists()
        assert not (branch_path / "verify_dp1.py").exists(), "Branched snapshot should match historical step state"
        assert (branch_path / ".broutegpt" / "events.jsonl").exists()

        # 4. Test decoupled rollback scopes: 'code' only vs 'chat' only
        (ws / "extra_after.txt").write_text("temporary file\n", encoding="utf-8")
        events_before_code_rb = len(EventStream(ws).events)
        rb_code = run_agent_cli(
            project_dir=str(ws),
            rollback_step_id=step_after_app,
            rollback_scope="code",
        )
        assert rb_code["scope"] == "code"
        assert rb_code["discarded_events"] == 0
        assert not (ws / "extra_after.txt").exists(), "code-only rollback must restore files"
        assert len(EventStream(ws).events) == events_before_code_rb, "code-only rollback must keep chat events intact"

        (ws / "keep_during_chat_rb.txt").write_text("survives chat rollback\n", encoding="utf-8")
        rb_chat = run_agent_cli(
            project_dir=str(ws),
            rollback_step_id=step_after_app,
            rollback_scope="chat",
        )
        assert rb_chat["scope"] == "chat"
        assert rb_chat["discarded_events"] > 0
        assert (ws / "keep_during_chat_rb.txt").exists(), "chat-only rollback must not delete workspace files"

        # 5. Verify CLI --help flags
        help_proc = subprocess.run(
            [sys.executable, str(ROOT / "broutegpt.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--scope" in help_proc.stdout and "--branch" in help_proc.stdout

    print("VERIFY_U10_ADVISOR_BRANCH: PASS")


if __name__ == "__main__":
    main()

