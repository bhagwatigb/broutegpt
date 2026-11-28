import ast
import inspect
import sys
from pathlib import Path

PLAY_DIR = Path.home() / "broutegpt_playground"
sys.path.insert(0, str(PLAY_DIR))

orch_file = PLAY_DIR / "broutegpt_agent" / "dp_orchestrator.py"
ast.parse(orch_file.read_text(encoding="utf-8"), filename="dp_orchestrator.py")

from broutegpt_agent.dp_orchestrator import DPOrchestrator

assert DPOrchestrator.__doc__ and "Decision-Point" in DPOrchestrator.__doc__
exec_src = inspect.getsource(DPOrchestrator._execute_single_dp)
assert "PROJECT RULES:" not in exec_src
assert "max_matches=" in exec_src and "max_results=" in exec_src
assert "consecutive_reads >= 6" in exec_src
assert exec_src.count("consecutive_reads = 0") >= 3

print("DP 2 SINGLE-DP VERIFICATION PASSED")

