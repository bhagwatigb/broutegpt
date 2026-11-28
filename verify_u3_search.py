import sys
import tempfile
from pathlib import Path

PLAY_DIR = Path.home() / "broutegpt_playground"
sys.path.insert(0, str(PLAY_DIR))

from broutegpt_agent.repo_map import RepoMapper
from broutegpt_agent.terminal_executor import TerminalExecutor

with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp)
    (ws / "alpha.py").write_text(
        "def helper_one():\n    before = 1\n    target_token = 'alpha_val'\n    after = 2\n    return target_token\n",
        encoding="utf-8",
    )
    (ws / "notes.md").write_text(
        "# Heading\nTARGET_TOKEN in markdown\n",
        encoding="utf-8",
    )

    ex = TerminalExecutor(ws, use_kernel_sandbox=False)
    mapper = RepoMapper(ex)

    res_ctx = mapper.grep_search("target_token", ".", context_lines=1)
    assert "before = 1" in res_ctx and "after = 2" in res_ctx, f"Missing context lines: {res_ctx}"

    res_glob = mapper.grep_search("target_token", ".", glob="*.py")
    assert "alpha.py" in res_glob and "notes.md" not in res_glob, f"Glob filter failed: {res_glob}"

    res_ci = mapper.grep_search("target_token", ".", case_insensitive=True)
    assert "notes.md" in res_ci, f"Case-insensitive search failed: {res_ci}"

    (ws / "many.py").write_text("\n".join(f"hit_{i} = 'dup'" for i in range(10)) + "\n", encoding="utf-8")
    res_lim = mapper.grep_search("= 'dup'", ".", max_matches=3, context_lines=0)
    assert "Showing 3 of 10" in res_lim, f"Missing truncation summary: {res_lim}"

print("DP 3 VERIFICATION PASSED")
print("  - Context lines (-C), glob filter, case_insensitive (-i), and truncation count verified")

