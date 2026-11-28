import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.condenser import (
    MEMORY_MAX_BYTES,
    MEMORY_MAX_LINES,
    RollingCondenser,
    compute_dynamic_token_budget,
)
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.event_stream import EventStream


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="u5_verify_") as tmp:
        ws = Path(tmp) / "proj"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "docs").mkdir(parents=True, exist_ok=True)

        # 1. Verify AGENTS.md + recursive @path imports + cycle safety
        (ws / "docs" / "nested.md").write_text(
            "Nested rule: never leave unverified files.\n@../AGENTS.md\n",
            encoding="utf-8",
        )
        (ws / "docs" / "extra_rule.md").write_text(
            "Imported rule: always validate output schemas before completing a DP.\n@nested.md\n",
            encoding="utf-8",
        )
        (ws / "AGENTS.md").write_text(
            "# Agent Rules\nFollow strict typing.\n@docs/extra_rule.md\n",
            encoding="utf-8",
        )
        (ws / "CLAUDE.md").write_text("Claude compatibility rule.\n", encoding="utf-8")

        orch = DPOrchestrator(ws)
        rules_loaded = orch._load_project_rules()
        assert "Follow strict typing." in rules_loaded, "AGENTS.md content missing"
        assert "Imported rule: always validate output schemas" in rules_loaded, "@path import failed"
        assert "Nested rule: never leave unverified files." in rules_loaded, "Nested @path import failed"
        assert "Claude compatibility rule." in rules_loaded, "CLAUDE.md content missing"

        # 2. Verify two-tier .broutegpt/MEMORY.md persistence and 200-line / 25 KB caps
        for i in range(260):
            orch.append_memory(f"Lesson #{i:03d}: deterministic check detail {'x' * 140}", category="lesson")

        mem_file = ws / ".broutegpt" / "MEMORY.md"
        assert mem_file.exists(), ".broutegpt/MEMORY.md should be created"
        raw_mem = mem_file.read_bytes()
        mem_lines = raw_mem.decode("utf-8", errors="replace").splitlines()
        assert len(mem_lines) <= MEMORY_MAX_LINES, f"MEMORY.md exceeded 200 lines: {len(mem_lines)}"
        assert len(raw_mem) <= MEMORY_MAX_BYTES, f"MEMORY.md exceeded 25 KB: {len(raw_mem)}"
        assert "Lesson #259:" in raw_mem.decode("utf-8"), "Most recent memory entries should be retained"

        rules_with_mem = orch._load_project_rules(include_memory=True)
        assert "Lesson #259:" in rules_with_mem, "MEMORY.md should be injected into project rules"

        # 3. Verify dynamic target_token_budget calculation from model input_limit
        assert compute_dynamic_token_budget(4096) < 4096
        assert compute_dynamic_token_budget(262144) > compute_dynamic_token_budget(32768)

        cond = RollingCondenser(ws / ".broutegpt")
        stream = EventStream(ws)
        stream.append("goal", "Build a pipeline with detailed observations.", dp_index=0)
        for k in range(1, 18):
            stream.append("action", f'{{"action": "read_file", "path": "mod_{k}.py"}}', dp_index=1)
            stream.append("observation", f"Observation {k}: " + ("detail " * 120), dp_index=1)

        small_res = cond.compile_history(stream, model_input_limit=4096, active_dp_index=1)
        large_res = cond.compile_history(stream, model_input_limit=262144, active_dp_index=1)
        assert small_res["target_token_budget"] < large_res["target_token_budget"]
        assert small_res["estimated_tokens"] <= small_res["target_token_budget"]
        assert small_res["estimated_tokens"] < 4096

    print("VERIFY_U5_MEMORY_BUDGET: PASS")


if __name__ == "__main__":
    main()

