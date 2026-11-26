import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .condenser import RollingCondenser
from .event_stream import EventStream


SHADOW_EXCLUDES = """.git/
.broutegpt/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
venv/
.venv/
node_modules/
"""


@dataclass
class CheckpointRecord:
    step_id: int
    commit_hash: str
    timestamp: float
    dp_index: int
    event_id: int
    label: str
    model: Optional[str] = None
    is_dp_verified: bool = False
    files_changed: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRecord":
        return cls(
            step_id=int(data["step_id"]),
            commit_hash=str(data["commit_hash"]),
            timestamp=float(data.get("timestamp", 0.0)),
            dp_index=int(data.get("dp_index", 0)),
            event_id=int(data.get("event_id", 0)),
            label=str(data.get("label", "")),
            model=data.get("model"),
            is_dp_verified=bool(data.get("is_dp_verified", False)),
            files_changed=str(data.get("files_changed", "")),
        )


class CheckpointManager:
    """
    Cline-inspired shadow Git checkpoint & time-travel manager:
      - Maintains a hidden shadow repository at <project_dir>/.broutegpt/shadow_git
        using --git-dir and --work-tree so every step (including untracked files)
        is snapshotted without touching or polluting the project's main .git history.
      - Creates clean commits & tags (`dp-<k>-verified`) in the main Git repo
        only when a Decision Point passes verification.
      - Supports step-level diff forensics and atomic rollback of both workspace
        files and EventStream/RollingCondenser context.
    """

    def __init__(self, project_dir: str | Path, auto_main_git: bool = True) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.project_root / ".broutegpt"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.shadow_git_dir = self.state_dir / "shadow_git"
        self.manifest_path = self.state_dir / "checkpoints.jsonl"
        self.auto_main_git = auto_main_git
        self._records: List[CheckpointRecord] = []

        self._init_shadow_repo()
        self._load_manifest()
        if not self._records:
            self.capture(dp_index=0, event_id=0, label="initial_state", model=None)

    def _run_git(
        self,
        args: List[str],
        use_shadow: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if use_shadow:
            cmd = [
                "git",
                f"--git-dir={self.shadow_git_dir}",
                f"--work-tree={self.project_root}",
                *args,
            ]
        else:
            cmd = ["git", "-C", str(self.project_root), *args]

        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if check and res.returncode != 0:
            raise RuntimeError(f"Git command failed ({' '.join(cmd)}): {res.stderr.strip() or res.stdout.strip()}")
        return res

    def _init_shadow_repo(self) -> None:
        if not (self.shadow_git_dir / "HEAD").exists():
            self.shadow_git_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "--bare", str(self.shadow_git_dir)], capture_output=True, text=True, check=True)
            self._run_git(["config", "core.bare", "false"])
            self._run_git(["config", "core.worktree", str(self.project_root)])
            self._run_git(["config", "user.name", "broutegpt-shadow"])
            self._run_git(["config", "user.email", "shadow@broutegpt.local"])

        info_dir = self.shadow_git_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "exclude").write_text(SHADOW_EXCLUDES, encoding="utf-8")

    def _load_manifest(self) -> None:
        self._records.clear()
        if not self.manifest_path.exists():
            return
        for raw_line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                self._records.append(CheckpointRecord.from_dict(json.loads(line)))
            except Exception:
                continue

    def _append_record(self, rec: CheckpointRecord) -> None:
        self._records.append(rec)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    @property
    def records(self) -> List[CheckpointRecord]:
        return list(self._records)

    def get_step(self, step_id: int) -> Optional[CheckpointRecord]:
        for rec in self._records:
            if rec.step_id == step_id:
                return rec
        return None

    def latest_verified_step(self) -> Optional[CheckpointRecord]:
        for rec in reversed(self._records):
            if rec.is_dp_verified:
                return rec
        return None

    def capture(
        self,
        dp_index: int,
        event_id: int,
        label: str,
        model: Optional[str] = None,
        is_dp_verified: bool = False,
    ) -> CheckpointRecord:
        """Captures an atomic snapshot of the entire workspace into the shadow Git repo."""
        self._run_git(["add", "-A"])
        msg = f"[Step {len(self._records) + 1} | DP {dp_index}] {label}"
        if model:
            msg += f" ({model})"
        self._run_git(["commit", "--allow-empty", "-m", msg])

        sha = self._run_git(["rev-parse", "HEAD"]).stdout.strip()
        stat_res = self._run_git(["show", "--stat", "--format=", sha], check=False)
        stat_lines = [ln.strip() for ln in stat_res.stdout.splitlines() if ln.strip()]
        summary = stat_lines[-1] if stat_lines else "no file delta"

        if is_dp_verified:
            self._run_git(["tag", "-f", f"dp-{dp_index}-verified", sha], check=False)

        rec = CheckpointRecord(
            step_id=len(self._records) + 1,
            commit_hash=sha,
            timestamp=time.time(),
            dp_index=dp_index,
            event_id=event_id,
            label=label,
            model=model,
            is_dp_verified=is_dp_verified,
            files_changed=summary,
        )
        self._append_record(rec)
        return rec

    def commit_verified_dp(
        self,
        dp_index: int,
        title: str,
        event_id: int,
        model: Optional[str] = None,
    ) -> CheckpointRecord:
        """
        Records a verified DP checkpoint in the shadow repo AND creates a clean
        milestone commit + tag in the main project Git repository.
        """
        rec = self.capture(
            dp_index=dp_index,
            event_id=event_id,
            label=f"dp_{dp_index}_verified: {title}",
            model=model,
            is_dp_verified=True,
        )

        if not self.auto_main_git:
            return rec

        main_git = self.project_root / ".git"
        if not main_git.exists():
            self._run_git(["init"], use_shadow=False)
            self._run_git(["config", "user.name", "broutegpt"], use_shadow=False)
            self._run_git(["config", "user.email", "agent@broutegpt.local"], use_shadow=False)

        gitignore = self.project_root / ".gitignore"
        existing_gi = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if ".broutegpt/" not in existing_gi:
            sep = "\n" if existing_gi and not existing_gi.endswith("\n") else ""
            gitignore.write_text(f"{existing_gi}{sep}.broutegpt/\n__pycache__/\n", encoding="utf-8")

        self._run_git(["add", "-A"], use_shadow=False)
        self._run_git(["commit", "--allow-empty", "-m", f"DP {dp_index} verified: {title}"], use_shadow=False)
        self._run_git(["tag", "-f", f"dp-{dp_index}-verified"], use_shadow=False, check=False)
        return rec

    def diff_between(self, from_step_id: int, to_step_id: Optional[int] = None) -> str:
        """Returns unified git diff between two shadow checkpoints (or from_step_id -> current working tree)."""
        rec_a = self.get_step(from_step_id)
        if rec_a is None:
            raise ValueError(f"Checkpoint step #{from_step_id} not found.")
        if to_step_id is not None:
            rec_b = self.get_step(to_step_id)
            if rec_b is None:
                raise ValueError(f"Checkpoint step #{to_step_id} not found.")
            res = self._run_git(["diff", rec_a.commit_hash, rec_b.commit_hash], check=False)
            return res.stdout.strip() or "(No file differences between selected steps)"

        # Diff against live working tree (stage temporarily via add -N or diff HEAD)
        res = self._run_git(["diff", rec_a.commit_hash], check=False)
        return res.stdout.strip() or "(Working tree matches checkpoint)"

    def inspect_step(self, step_id: int, stream: Optional[EventStream] = None) -> str:
        """Forensic report for a specific step: model used, files touched, unified diff, and linked event."""
        rec = self.get_step(step_id)
        if rec is None:
            raise ValueError(f"Checkpoint step #{step_id} not found.")

        prev_rec = self.get_step(step_id - 1)
        if prev_rec is not None:
            diff_txt = self.diff_between(prev_rec.step_id, rec.step_id)
        else:
            diff_txt = self._run_git(["show", "--format=", rec.commit_hash], check=False).stdout.strip()

        lines = [
            f"=== Checkpoint Step #{rec.step_id} Forensics ===",
            f"  DP Index   : {rec.dp_index}",
            f"  Event ID   : #{rec.event_id}",
            f"  Label      : {rec.label}",
            f"  Model      : {rec.model or 'system'}",
            f"  Verified   : {'YES' if rec.is_dp_verified else 'No'}",
            f"  Shadow SHA : {rec.commit_hash[:12]}",
            f"  Delta      : {rec.files_changed}",
        ]
        if stream is not None and rec.event_id > 0:
            ev = stream.get(rec.event_id)
            if ev is not None:
                lines.append(f"  Event Kind : {ev.kind}")
                lines.append(f"  Event Body : {ev.content[:400]}")
        lines.append("\n--- Step Unified Diff ---")
        lines.append(diff_txt[:6000] if diff_txt else "(Empty diff)")
        return "\n".join(lines)

    def format_timeline(self) -> str:
        """Renders a readable forensic timeline of all checkpoints to pinpoint regressions/hallucinations."""
        if not self._records:
            return "No checkpoints recorded yet."
        header = f"{'Step':<5} {'DP':<4} {'Evt':<5} {'Verified':<9} {'Model':<32} {'Label / Delta'}"
        lines = [header, "-" * 95]
        for rec in self._records:
            ver = "VERIFIED" if rec.is_dp_verified else "-"
            mod = (rec.model or "system")[:31]
            desc = f"{rec.label} [{rec.files_changed}]"
            lines.append(f"{rec.step_id:<5} {rec.dp_index:<4} {rec.event_id:<5} {ver:<9} {mod:<32} {desc}")
        return "\n".join(lines)

    def rollback_to_step(
        self,
        step_id: int,
        stream: Optional[EventStream] = None,
        condenser: Optional[RollingCondenser] = None,
        restore_context: bool = True,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomically restores all workspace files (and removes untracked files created
        after `step_id`) to the exact snapshot of `step_id`. When `restore_context=True`,
        also rewinds `EventStream` and invalidates stale `RollingCondenser` cache entries.
        """
        target = self.get_step(step_id)
        if target is None:
            raise ValueError(f"Cannot rollback: checkpoint step #{step_id} does not exist.")

        self._run_git(["read-tree", "--reset", "-u", target.commit_hash])
        self._run_git(["clean", "-fd"])


        why = reason or f"Manual/autonomous rollback to step #{step_id} ({target.label})"
        discarded_events = 0
        if restore_context and stream is not None:
            discarded_events = stream.rewind_to(target.event_id, reason=why)
            if condenser is not None:
                condenser.invalidate_after(target.event_id)

        post_event_id = stream.events[-1].id if (stream is not None and stream.events) else target.event_id
        rollback_rec = self.capture(
            dp_index=target.dp_index,
            event_id=post_event_id,
            label=f"rollback_to_step_{step_id}",
            model=None,
            is_dp_verified=target.is_dp_verified,
        )
        return {
            "restored_step_id": step_id,
            "restored_commit": target.commit_hash,
            "restored_dp_index": target.dp_index,
            "restored_event_id": target.event_id,
            "discarded_events": discarded_events,
            "new_checkpoint_step_id": rollback_rec.step_id,
        }

