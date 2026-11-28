import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.code_editor import validate_action_json
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.subagent import ExploreSubagent, _validate_subagent_json


def main() -> None:
    # 1. Verify read-only action guard in subagent validator
    assert _validate_subagent_json('{"thought":"read","action":"read_file","path":"a.py"}') is None
    assert _validate_subagent_json('{"thought":"grep","action":"grep_search","pattern":"def "}') is None
    assert _validate_subagent_json('{"thought":"write","action":"write_file","path":"a.py","content":"x"}') is not None
    assert _validate_subagent_json('{"thought":"shell","action":"run_shell","command":"ls"}') is not None
    assert validate_action_json('{"thought":"explore","action":"spawn_subagent","task":"Inspect repo"}') is None

    with tempfile.TemporaryDirectory(prefix="u11_verify_") as tmp:
        ws = Path(tmp) / "proj"
        (ws / "pkg").mkdir(parents=True, exist_ok=True)
        (ws / "pkg" / "service.py").write_text(
            "def compute_tax(amount: float) -> float:\n    SECRET_MARKER = 'RAW_LINE_CONTENT_999'\n    return amount * 0.18\n",
            encoding="utf-8",
        )

        # 2. Test ExploreSubagent directly with isolated stream
        mock_router = MagicMock()
        mock_router.router = MagicMock()
        mock_router.hook_engine = None
        mock_router.call_log = []
        mock_router.get_active_input_limit.return_value = 65536

        sub_turns = iter([
            ('{"thought":"search tax","action":"grep_search","pattern":"compute_tax","path":"pkg"}', "groq:llama-3.1-8b-instant"),
            ('{"thought":"read service","action":"read_file","path":"pkg/service.py","start_line":1,"end_line":10}', "groq:llama-3.1-8b-instant"),
            ('{"thought":"done","action":"subagent_done","summary":"Found compute_tax in pkg/service.py:1 applying 18% tax."}', "groq:llama-3.1-8b-instant"),
        ])
        mock_router.query_autonomous.side_effect = lambda *a, **kw: next(sub_turns)

        orch = DPOrchestrator(ws, router=mock_router, auto_main_git=False)
        explorer = ExploreSubagent(mapper=orch.mapper, router=mock_router, max_turns=5)
        rep = explorer.run("Locate tax computation logic", scope_path="pkg")
        assert rep.ok and rep.turns_used == 3
        assert "pkg/service.py:1" in rep.summary
        assert not (ws / ".broutegpt" / "subagent_ephemeral.jsonl").exists()

        # 3. Test spawn_subagent inside DPOrchestrator: parent EventStream stores ONLY summary, not raw file dump
        plan_json = json.dumps({
            "project_name": "subagent_demo",
            "overview": "Subagent isolation test",
            "entrypoint": "pkg/service.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Explore and verify",
                    "description": "Use spawn_subagent then write verify_dp1.py",
                    "deliverables": ["verify_dp1.py"],
                    "verification_criteria": "verify_dp1.py exits 0",
                }
            ],
        })
        main_turns = iter([
            plan_json,
            # Parent turn 1: spawn_subagent
            json.dumps({"thought": "explore service", "action": "spawn_subagent", "task": "Inspect pkg/service.py", "path": "pkg"}),
            # Subagent turn 1: read_file (runs inside isolated stream)
            json.dumps({"thought": "read file", "action": "read_file", "path": "pkg/service.py", "start_line": 1, "end_line": 10}),
            # Subagent turn 2: subagent_done
            json.dumps({"thought": "finish explore", "action": "subagent_done", "summary": "pkg/service.py defines compute_tax(amount) -> float."}),
            # Parent turn 2: write verify_dp1.py and complete DP 1
            json.dumps({
                "thought": "write verification",
                "actions": [
                    {"action": "write_file", "path": "verify_dp1.py", "content": "from pkg.service import compute_tax\nassert round(compute_tax(100), 2) == 18.0\n"},
                    {"action": "dp_complete", "summary": "verified"},
                ],
            }),
        ])
        mock_router.query_autonomous.side_effect = lambda *a, **kw: (next(main_turns), "groq:llama-3.1-8b-instant")

        orch2 = DPOrchestrator(ws, router=mock_router, auto_main_git=False)
        orch2.run("Test isolated explore subagent")

        parent_events_text = "\n".join(e.content for e in orch2.stream.events)
        assert "SUBAGENT SUMMARY (groq:llama-3.1-8b-instant, 2 turn(s))" in parent_events_text
        assert "pkg/service.py defines compute_tax(amount) -> float." in parent_events_text
        assert "RAW_LINE_CONTENT_999" not in parent_events_text, "Raw subagent file read leaked into parent EventStream"

    print("VERIFY_U11_SUBAGENT: PASS")


if __name__ == "__main__":
    main()

