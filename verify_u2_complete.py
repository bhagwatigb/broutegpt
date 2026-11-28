import ast
import inspect
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PLAY_DIR = Path.home() / "broutegpt_playground"
sys.path.insert(0, str(PLAY_DIR))

core_files = [PLAY_DIR / "broutegpt.py", *(PLAY_DIR / "broutegpt_agent").glob("*.py")]
assert len(core_files) >= 11, f"Expected at least 11 core .py files, found {len(core_files)}"
for pyf in core_files:
    ast.parse(pyf.read_text(encoding="utf-8"), filename=str(pyf.relative_to(PLAY_DIR)))

import broutegpt
from broutegpt_agent.checkpoint_manager import CheckpointManager
from broutegpt_agent.condenser import RollingCondenser
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.event_stream import EventStream
from broutegpt_agent.router_adapter import CloudFirstRouter

assert DPOrchestrator.__doc__ and "Decision-Point" in DPOrchestrator.__doc__, "Missing DPOrchestrator.__doc__"
exec_src = inspect.getsource(DPOrchestrator._execute_single_dp)
assert "PROJECT RULES:" not in exec_src, "Duplicate PROJECT RULES in turn_prompt"
assert "max_matches=" in exec_src and "max_results=" in exec_src, "Search limits not forwarded"
assert "consecutive_reads >= 6" in exec_src, "Missing read-loop circuit breaker"
assert exec_src.count("consecutive_reads = 0") >= 3, "Missing consecutive_reads reset on edits"

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    stream = EventStream(tmp_path)
    long_plan = "PLAN_MARKER_START " + ("A" * 5000) + " PLAN_MARKER_END"
    stream.append("goal", "Upgrade broutegpt", dp_index=0)
    stream.append("rules", "ALWAYS_PINNED_RULE", dp_index=0)
    stream.append("plan", long_plan, dp_index=0)
    for i in range(14):
        stream.append("observation", f"turn_{i}_" + ("B" * 800), dp_index=1)
    cond = RollingCondenser(tmp_path / ".broutegpt", keep_tail_events=4, condense_batch_min=2)
    assert cond.keep_tail_events == 4
    compiled = cond.compile_history(stream, target_token_budget=1800, active_dp_index=1, router=None)
    joined = "\n".join(m["content"] for m in compiled["history"])
    assert "PLAN_MARKER_START" in joined and "PLAN_MARKER_END" in joined, "Pinned plan head_event clipped"
    assert "ALWAYS_PINNED_RULE" in joined, "Pinned rules head_event missing"

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    ckpt = CheckpointManager(tmp_path, auto_main_git=False)
    plan_file = tmp_path / ".broutegpt" / "dp_plan.json"
    plan_file.write_text(json.dumps({
        "project_name": "demo",
        "dps": [
            {"id": 1, "title": "DP1", "status": "verified"},
            {"id": 2, "title": "DP2", "status": "verified"},
        ],
    }), encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    ckpt.commit_verified_dp(dp_index=1, title="DP1", event_id=1)
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    ckpt.commit_verified_dp(dp_index=2, title="DP2", event_id=2)

    ckpt.rollback_to_step(2, restore_context=False)
    synced = json.loads(plan_file.read_text(encoding="utf-8"))
    statuses = {d["id"]: d["status"] for d in synced["dps"]}
    assert statuses == {1: "verified", 2: "pending"}, f"Unexpected dp_plan statuses: {statuses}"

router = broutegpt.FreeLLMRouter.__new__(broutegpt.FreeLLMRouter)
router.timeout = 10
router.last_call_usage = {}
router._cooldown_until = {}
router.models_by_option = {
    "1": [],
    "3": [
        {"id": "dead-model", "provider": "groq", "input_limit": 8192, "output_limit": 4096, "chat": True, "est_tps": 800.0},
        {"id": "llama-3.3-70b-versatile", "provider": "groq", "input_limit": 32768, "output_limit": 8192, "chat": True, "est_tps": 320.0},
    ],
}
router.role_config = {}
router.selected_option = None
router.custom_model = None
router._save_cache = MagicMock()

ok_resp = MagicMock()
ok_resp.raise_for_status.return_value = None
ok_resp.json.return_value = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 312, "completion_tokens": 64},
}

def side_effect_post(*args, **kwargs):
    payload = kwargs.get("json") or {}
    if payload.get("model") == "dead-model":
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = RuntimeError("404 Not Found for model dead-model")
        return err_resp
    return ok_resp

with patch("requests.post", side_effect=side_effect_post):
    cfr = CloudFirstRouter(router=router)
    out, used = cfr.query_autonomous("hi", tier="speed")
    assert out == "ok" and used == "groq:llama-3.3-70b-versatile"
    assert all(m["id"] != "dead-model" for m in router.models_by_option["3"]), "Dead 404 model not evicted"
    ok_logs = [e for e in cfr.call_log if e.get("status") == "ok"]
    assert ok_logs and ok_logs[-1]["prompt_tokens"] == 312 and ok_logs[-1]["completion_tokens"] == 64
    assert ok_logs[-1].get("token_source") == "api"

print("DP 2 COMPLETE VERIFICATION PASSED")
print(f"  - Checked {len(core_files)} core Python files for valid AST")
print("  - Verified dp_orchestrator.py, condenser.py, checkpoint_manager.py, broutegpt.py, router_adapter.py")

