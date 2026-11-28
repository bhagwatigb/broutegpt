import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

os.environ["BROUTEGPT_NO_SANDBOX"] = "1"

from broutegpt_agent.checkpoint_manager import CheckpointManager
from broutegpt_agent.dp_orchestrator import DPOrchestrator


class ThreadSafeMockRouter:
    """Deterministic, thread-safe mock router that records concurrent execution overlap."""

    def __init__(self, mode: str = "diamond") -> None:
        self.mode = mode
        self.router = None
        self.hook_engine = None
        self.call_log = []
        self._lock = threading.Lock()
        self.active_workers = 0
        self.max_concurrent_workers = 0

    def query_autonomous(self, prompt: str, **kwargs):
        with self._lock:
            self.active_workers += 1
            if self.active_workers > self.max_concurrent_workers:
                self.max_concurrent_workers = self.active_workers

        try:
            # Small sleep so parallel worktree threads overlap observably
            time.sleep(0.18)
            m = re.search(r"ACTIVE DECISION POINT:\s*DP\s*(\d+)", prompt)
            dp_id = int(m.group(1)) if m else 1

            if self.mode == "diamond":
                reply = self._diamond_reply(dp_id)
            elif self.mode == "conflict":
                reply = self._conflict_reply(dp_id, prompt, kwargs.get("history") or [])
            else:
                raise ValueError(f"Unknown mock mode: {self.mode}")

            with self._lock:
                self.call_log.append({
                    "status": "ok",
                    "prompt_tokens": 120,
                    "completion_tokens": 80,
                })
            return reply, f"mock-worker-dp{dp_id}"
        finally:
            with self._lock:
                self.active_workers -= 1

    def _diamond_reply(self, dp_id: int) -> str:
        if dp_id == 1:
            return json.dumps({
                "thought": "Building alpha.py and verify_dp1.py in isolated worktree.",
                "actions": [
                    {
                        "action": "write_file",
                        "path": "alpha.py",
                        "content": "def get_alpha() -> int:\n    return 21\n",
                    },
                    {
                        "action": "write_file",
                        "path": "verify_dp1.py",
                        "content": "from alpha import get_alpha\nassert get_alpha() == 21\nprint('DP1_OK')\n",
                    },
                    {"action": "run_shell", "command": "python3 verify_dp1.py"},
                    {"action": "dp_complete", "summary": "Built alpha.py"},
                ],
            })
        if dp_id == 2:
            return json.dumps({
                "thought": "Building beta.py and verify_dp2.py in isolated worktree.",
                "actions": [
                    {
                        "action": "write_file",
                        "path": "beta.py",
                        "content": "def get_beta() -> int:\n    return 2\n",
                    },
                    {
                        "action": "write_file",
                        "path": "verify_dp2.py",
                        "content": "from beta import get_beta\nassert get_beta() == 2\nprint('DP2_OK')\n",
                    },
                    {"action": "run_shell", "command": "python3 verify_dp2.py"},
                    {"action": "dp_complete", "summary": "Built beta.py"},
                ],
            })
        return json.dumps({
            "thought": "Integrating alpha.py and beta.py in DP 3 after wave 1 merged.",
            "actions": [
                {
                    "action": "write_file",
                    "path": "combined.py",
                    "content": (
                        "from alpha import get_alpha\n"
                        "from beta import get_beta\n\n"
                        "def compute_total() -> int:\n"
                        "    return get_alpha() * get_beta()\n"
                    ),
                },
                {
                    "action": "write_file",
                    "path": "verify_dp3.py",
                    "content": "from combined import compute_total\nassert compute_total() == 42\nprint('DP3_OK')\n",
                },
                {"action": "run_shell", "command": "python3 verify_dp3.py"},
                {"action": "dp_complete", "summary": "Integrated alpha + beta = 42"},
            ],
        })

    def _conflict_reply(self, dp_id: int, prompt: str, history: list) -> str:
        # Both DP 1 and DP 2 modify the exact same line in shared_cfg.py during the parallel wave;
        # when DP 2 falls back sequentially and sees verify_dp1.py in regression history, it preserves VALUE_A = 10.
        saw_dp1_regression = any("verify_dp1.py" in str(msg) for msg in (history or []))
        if dp_id == 1:
            content = "VALUE_A = 10\nVALUE_B = 0\n"
            verify_code = "import shared_cfg\nassert shared_cfg.VALUE_A == 10\n"
        else:
            val_a = 10 if saw_dp1_regression else 999
            content = f"VALUE_A = {val_a}\nVALUE_B = 20\n"
            verify_code = "import shared_cfg\nassert shared_cfg.VALUE_B == 20\n"

        return json.dumps({
            "thought": f"Writing shared_cfg.py for DP {dp_id}.",
            "actions": [
                {"action": "write_file", "path": "shared_cfg.py", "content": content},
                {"action": "write_file", "path": f"verify_dp{dp_id}.py", "content": verify_code},
                {"action": "run_shell", "command": f"python3 verify_dp{dp_id}.py"},
                {"action": "dp_complete", "summary": f"Updated shared_cfg for DP {dp_id}"},
            ],
        })


def test_wave_scheduler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orch = DPOrchestrator(tmp, router=ThreadSafeMockRouter(), auto_main_git=False)
        diamond_dps = [
            {"id": 1, "title": "A", "depends_on": []},
            {"id": 2, "title": "B", "depends_on": []},
            {"id": 3, "title": "C", "depends_on": [1, 2]},
            {"id": 4, "title": "D", "depends_on": [3]},
        ]
        waves = orch.compute_execution_waves(diamond_dps)
        assert [[d["id"] for d in w] for w in waves] == [[1, 2], [3], [4]], f"Unexpected waves: {waves}"

        legacy_dps = [{"id": 1, "title": "Seq1"}, {"id": 2, "title": "Seq2"}]
        seq_waves = orch.compute_execution_waves(legacy_dps)
        assert [[d["id"] for d in w] for w in seq_waves] == [[1], [2]], f"Legacy waves broken: {seq_waves}"

        try:
            orch.compute_execution_waves([
                {"id": 1, "title": "C1", "depends_on": [2]},
                {"id": 2, "title": "C2", "depends_on": [1]},
            ])
            raise AssertionError("Expected cycle detection ValueError")
        except ValueError as exc:
            assert "Cycle detected" in str(exc)

        try:
            orch.compute_execution_waves([{"id": 1, "title": "Bad", "depends_on": [99]}])
            raise AssertionError("Expected unknown dependency ValueError")
        except ValueError as exc:
            assert "unknown DP ID" in str(exc)
    print("[PASS] 1. DAG topological wave scheduler, legacy fallback, and cycle/ID validation")


def test_worktree_checkpoint_primitives() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "base.txt").write_text("line1\n", encoding="utf-8")
        cm = CheckpointManager(root, auto_main_git=False)
        base_ckpt = cm.capture(dp_index=0, event_id=0, label="base_with_file")

        wt1 = cm.create_dp_worktree(1, base_commit=base_ckpt.commit_hash)
        wt2 = cm.create_dp_worktree(2, base_commit=base_ckpt.commit_hash)
        wt1_path = Path(wt1["worktree_dir"])
        wt2_path = Path(wt2["worktree_dir"])
        assert wt1_path.is_dir() and wt2_path.is_dir()

        (wt1_path / "only_wt1.py").write_text("X = 1\n", encoding="utf-8")
        (wt2_path / "only_wt2.py").write_text("Y = 2\n", encoding="utf-8")

        m1 = cm.merge_dp_worktree(1, wt1_path, title="WT1")
        cm.cleanup_dp_worktree(1, wt1_path)
        m2 = cm.merge_dp_worktree(2, wt2_path, title="WT2")
        cm.cleanup_dp_worktree(2, wt2_path)

        assert m1["merged"] is True and m2["merged"] is True
        assert (root / "only_wt1.py").read_text(encoding="utf-8") == "X = 1\n"
        assert (root / "only_wt2.py").read_text(encoding="utf-8") == "Y = 2\n"
        assert not wt1_path.exists() and not wt2_path.exists()

        # Verify conflict detection aborts cleanly without leaving conflict markers
        conflict_base = cm.capture(dp_index=0, event_id=0, label="before_conflict")
        c1 = cm.create_dp_worktree(3, base_commit=conflict_base.commit_hash)
        c2 = cm.create_dp_worktree(4, base_commit=conflict_base.commit_hash)
        (Path(c1["worktree_dir"]) / "base.txt").write_text("branch_3_edit\n", encoding="utf-8")
        (Path(c2["worktree_dir"]) / "base.txt").write_text("branch_4_edit\n", encoding="utf-8")

        res_c1 = cm.merge_dp_worktree(3, c1["worktree_dir"], title="C1")
        cm.cleanup_dp_worktree(3, c1["worktree_dir"])
        res_c2 = cm.merge_dp_worktree(4, c2["worktree_dir"], title="C2")
        cm.cleanup_dp_worktree(4, c2["worktree_dir"])

        assert res_c1["merged"] is True
        assert res_c2["merged"] is False and res_c2["conflict"] is True
        assert "<<<<<<<" not in (root / "base.txt").read_text(encoding="utf-8")
    print("[PASS] 2. CheckpointManager worktree isolation, 3-way merge, conflict abort, and cleanup")


def test_end_to_end_parallel_dag_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state_dir = root / ".broutegpt"
        state_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "project_name": "diamond_parallel_demo",
            "overview": "Parallel alpha & beta modules merged before combined DP 3.",
            "entrypoint": "combined.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Build Alpha",
                    "depends_on": [],
                    "description": "Create alpha.py",
                    "deliverables": ["alpha.py", "verify_dp1.py"],
                    "verification_criteria": "get_alpha() == 21",
                    "status": "pending",
                },
                {
                    "id": 2,
                    "title": "Build Beta",
                    "depends_on": [],
                    "description": "Create beta.py",
                    "deliverables": ["beta.py", "verify_dp2.py"],
                    "verification_criteria": "get_beta() == 2",
                    "status": "pending",
                },
                {
                    "id": 3,
                    "title": "Combine Alpha and Beta",
                    "depends_on": [1, 2],
                    "description": "Create combined.py using alpha and beta",
                    "deliverables": ["combined.py", "verify_dp3.py"],
                    "verification_criteria": "compute_total() == 42",
                    "status": "pending",
                },
            ],
        }
        (state_dir / "dp_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        mock_router = ThreadSafeMockRouter(mode="diamond")
        orch = DPOrchestrator(root, router=mock_router, auto_main_git=True)
        report = orch.run(goal="Build diamond parallel project", resume=True)

        assert mock_router.max_concurrent_workers >= 2, (
            f"Expected concurrent worktree execution (>=2 active workers), got {mock_router.max_concurrent_workers}"
        )
        assert report["total_dps"] == 3
        assert len(report["results"]) == 3
        modes = {r["dp_id"]: r.get("execution_mode", "sequential") for r in report["results"]}
        assert modes[1] == "parallel_worktree"
        assert modes[2] == "parallel_worktree"

        assert (root / "alpha.py").is_file()
        assert (root / "beta.py").is_file()
        assert (root / "combined.py").is_file()

        wt_root = state_dir / "worktrees"
        remaining_wts = list(wt_root.iterdir()) if wt_root.exists() else []
        assert not remaining_wts, f"Expected all temporary worktrees cleaned up, found: {remaining_wts}"

        # Verify git tags dp-1-verified, dp-2-verified, dp-3-verified exist in main .git
        tags_out = subprocess.run(
            ["git", "-C", str(root), "tag"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        for expected_tag in ("dp-1-verified", "dp-2-verified", "dp-3-verified"):
            assert expected_tag in tags_out, f"Missing git tag {expected_tag}: {tags_out}"
    print("[PASS] 3. End-to-end parallel DAG worktree execution, merge, regression suite, and git tags")


def test_conflict_fallback_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state_dir = root / ".broutegpt"
        state_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "project_name": "conflict_fallback_demo",
            "overview": "Two parallel DPs editing the same line trigger clean sequential fallback.",
            "entrypoint": "shared_cfg.py",
            "dps": [
                {
                    "id": 1,
                    "title": "Set VALUE_A",
                    "depends_on": [],
                    "description": "Set VALUE_A = 10 in shared_cfg.py",
                    "deliverables": ["shared_cfg.py", "verify_dp1.py"],
                    "verification_criteria": "VALUE_A == 10",
                    "status": "pending",
                },
                {
                    "id": 2,
                    "title": "Set VALUE_B",
                    "depends_on": [],
                    "description": "Set VALUE_B = 20 in shared_cfg.py",
                    "deliverables": ["shared_cfg.py", "verify_dp2.py"],
                    "verification_criteria": "VALUE_B == 20",
                    "status": "pending",
                },
            ],
        }
        (state_dir / "dp_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        mock_router = ThreadSafeMockRouter(mode="conflict")
        orch = DPOrchestrator(root, router=mock_router, auto_main_git=False)
        report = orch.run(goal="Test merge conflict fallback", resume=True)

        assert len(report["results"]) == 2
        res_by_id = {r["dp_id"]: r for r in report["results"]}
        assert res_by_id[1]["execution_mode"] == "parallel_worktree"
        assert res_by_id[2]["execution_mode"] == "sequential_conflict_fallback"
        cfg_txt = (root / "shared_cfg.py").read_text(encoding="utf-8")
        assert "VALUE_A = 10" in cfg_txt and "VALUE_B = 20" in cfg_txt
    print("[PASS] 4. Graceful sequential fallback when parallel worktree branches conflict")


if __name__ == "__main__":
    test_wave_scheduler()
    test_worktree_checkpoint_primitives()
    test_end_to_end_parallel_dag_execution()
    test_conflict_fallback_execution()
    print("\nALL DP 12 PARALLEL DAG WORKTREE CHECKS PASSED!")

