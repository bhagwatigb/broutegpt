import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .terminal_executor import _scrub_child_env


LIFECYCLE_EVENTS = (
    "SessionStart",
    "PreToolUse",
    "PostToolUse",
    "PostToolBatch",
    "PreCompact",
    "PreModelSwitch",
    "DPVerifyFailed",
    "Stop",
)


@dataclass
class HookDispatchResult:
    event: str
    executed_count: int = 0
    blocked: bool = False
    block_reason: str = ""
    additional_context: str = ""
    warnings: List[str] = field(default_factory=list)


class HookEngine:
    """
    Deterministic lifecycle hook runner backed by `<project_root>/.broutegpt/hooks.json`
    (and optional `~/.broutegpt/hooks.json`).
      - Exit code 2 blocks the operation and surfaces stderr/stdout as block_reason.
      - Non-empty stdout from non-blocking hooks is returned as additionalContext.
    """

    def __init__(self, project_dir: str | Path, default_timeout: int = 20) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.state_dir = self.project_root / ".broutegpt"
        self.hooks_path = self.state_dir / "hooks.json"
        self.user_hooks_path = Path.home() / ".broutegpt" / "hooks.json"
        self.default_timeout = int(default_timeout)

    def _load_event_specs(self, event_name: str) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for path in (self.user_hooks_path, self.hooks_path):
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            container = raw.get("hooks") if isinstance(raw.get("hooks"), dict) else raw
            entries = container.get(event_name)
            if not isinstance(entries, list):
                continue
            for item in entries:
                if isinstance(item, str) and item.strip():
                    specs.append({"command": item.strip(), "matcher": "", "timeout": self.default_timeout})
                elif isinstance(item, dict) and str(item.get("command") or "").strip():
                    specs.append({
                        "command": str(item["command"]).strip(),
                        "matcher": str(item.get("matcher") or "").strip(),
                        "timeout": int(item.get("timeout") or self.default_timeout),
                    })
        return specs

    def _matches(self, matcher: str, payload: Dict[str, Any]) -> bool:
        if not matcher or matcher == "*":
            return True
        candidates = [
            str(payload.get("tool") or ""),
            str(payload.get("action") or ""),
            str(payload.get("stage") or ""),
            str(payload.get("to_model") or ""),
            str(payload.get("from_model") or ""),
            str(payload.get("dp_id") or ""),
        ]
        targets = [c for c in candidates if c]
        if not targets:
            return False
        for target in targets:
            try:
                if re.search(matcher, target):
                    return True
            except re.error:
                pass
            if fnmatch.fnmatch(target, matcher) or matcher in target:
                return True
        return False

    def dispatch(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> HookDispatchResult:
        data = dict(payload or {})
        data.setdefault("event", event_name)
        data.setdefault("project_dir", str(self.project_root))

        specs = self._load_event_specs(event_name)
        if not specs:
            return HookDispatchResult(event=event_name)

        shell_bin = "/bin/zsh" if Path("/bin/zsh").exists() else (shutil.which("bash") or "/bin/sh")
        payload_json = json.dumps(data, ensure_ascii=False, default=str)

        env = _scrub_child_env(os.environ.copy())
        env["PROJECT_WORKSPACE"] = str(self.project_root)
        env["BROUTEGPT_HOOK_EVENT"] = event_name
        env["BROUTEGPT_HOOK_PAYLOAD"] = payload_json[:32000]
        if data.get("tool") or data.get("action"):
            env["BROUTEGPT_HOOK_TOOL"] = str(data.get("tool") or data.get("action"))

        context_chunks: List[str] = []
        warnings: List[str] = []
        executed = 0

        for spec in specs:
            if not self._matches(spec["matcher"], data):
                continue
            executed += 1
            try:
                proc = subprocess.run(
                    [shell_bin, "-lc", spec["command"]],
                    cwd=str(self.project_root),
                    input=payload_json,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=spec["timeout"],
                    env=env,
                )
            except subprocess.TimeoutExpired:
                warnings.append(f"Hook '{spec['command']}' timed out after {spec['timeout']}s")
                continue
            except Exception as exc:
                warnings.append(f"Hook '{spec['command']}' failed to start: {exc}")
                continue

            out_txt = (proc.stdout or "").strip()
            err_txt = (proc.stderr or "").strip()

            if proc.returncode == 2:
                reason = err_txt or out_txt or f"Blocked by {event_name} hook (exit code 2)"
                return HookDispatchResult(
                    event=event_name,
                    executed_count=executed,
                    blocked=True,
                    block_reason=reason,
                    additional_context="\n".join(context_chunks).strip(),
                    warnings=warnings,
                )

            if out_txt:
                context_chunks.append(out_txt)
            if proc.returncode != 0:
                warnings.append(
                    f"Hook '{spec['command']}' exited with code {proc.returncode}: {err_txt or out_txt}"
                )

        return HookDispatchResult(
            event=event_name,
            executed_count=executed,
            blocked=False,
            block_reason="",
            additional_context="\n".join(context_chunks).strip(),
            warnings=warnings,
        )

