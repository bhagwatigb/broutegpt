import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


BLOCKED_COMMAND_PATTERNS = [
    r"(^|\s)sudo(\s|$)",
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/|~|\$HOME|\.\.)",
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\$HOME|\.\.)",
    r"(^|\s)(mkfs|fdisk|diskutil\s+erase|dd\s+if=|shutdown|reboot|halt|launchctl)(\s|$)",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"(>|>>)\s*(~|/Users/[^/]+/\.(zshrc|zprofile|bashrc|bash_profile|ssh))",
]

ALLOWED_SYSTEM_PREFIXES = (
    "/usr/",
    "/bin/",
    "/sbin/",
    "/opt/homebrew/",
    "/Library/",
    "/System/Library/",
    "/dev/",
    "/private/tmp/",
    "/tmp/",
    "/var/folders/",
)


def _truncate_middle(text: str, max_chars: int = 16000) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.45)
    tail = int(max_chars * 0.50)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [TRUNCATED {omitted:,} CHARS] ...\n\n{text[-tail:]}"


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def format_for_llm(self, max_chars: int = 14000) -> str:
        status = "TIMEOUT" if self.timed_out else f"EXIT_CODE={self.exit_code}"
        combined = f"$ {self.command}\n[{status} in {self.duration_sec:.2f}s]\n"
        if self.stdout.strip():
            combined += f"--- STDOUT ---\n{self.stdout.strip()}\n"
        if self.stderr.strip():
            combined += f"--- STDERR ---\n{self.stderr.strip()}\n"
        return _truncate_middle(combined, max_chars=max_chars)


class TerminalExecutor:
    """
    Workspace-jailed macOS Zsh executor with process-group timeout termination
    and path/command boundary enforcement.
    """

    def __init__(self, workspace_dir: str | Path, default_timeout: int = 90) -> None:
        self.workspace_root = Path(workspace_dir).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout

    def resolve_safe_path(self, target_path: str | Path) -> Path:
        raw = Path(target_path)
        candidate = (raw if raw.is_absolute() else (self.workspace_root / raw)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                f"Workspace jail violation: '{target_path}' resolves outside '{self.workspace_root}'"
            ) from exc
        return candidate

    def validate_command(self, command: str) -> None:
        cmd = command.strip()
        if not cmd:
            raise ValueError("Command cannot be empty.")

        for pat in BLOCKED_COMMAND_PATTERNS:
            if re.search(pat, cmd, re.IGNORECASE):
                raise PermissionError(f"Blocked destructive/system-level command pattern: {cmd}")

        # Block explicit parent-directory traversal escapes
        if re.search(r"(^|[\s/'\"])(\.\./){2,}", cmd):
            raise PermissionError(f"Blocked multi-level parent traversal in command: {cmd}")

        # Check explicit absolute file paths referenced in command tokens
        ws_str = str(self.workspace_root)
        for token in re.findall(r"(?:^|[\s=:'\"])(/[^\s'\"`;|&<>]+)", cmd):
            if token == "/":
                raise PermissionError("Blocked root path '/' reference in shell command.")
            if token.startswith(ALLOWED_SYSTEM_PREFIXES):
                continue
            resolved_tok = str(Path(token).resolve())
            if not (resolved_tok == ws_str or resolved_tok.startswith(ws_str + os.sep)):
                raise PermissionError(
                    f"Workspace jail violation in command: '{token}' is outside '{ws_str}'"
                )

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        env_extra: Optional[Dict[str, str]] = None,
        max_output_chars: int = 16000,
    ) -> CommandResult:
        self.validate_command(command)
        eff_timeout = timeout or self.default_timeout

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PROJECT_WORKSPACE"] = str(self.workspace_root)
        if env_extra:
            env.update(env_extra)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            ["/bin/zsh", "-lc", command],
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            env=env,
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=eff_timeout)
            exit_code = int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -9
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", f"Command timed out after {eff_timeout}s and process group was killed."

        duration = round(time.perf_counter() - t0, 3)
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=_truncate_middle(stdout or "", max_chars=max_output_chars),
            stderr=_truncate_middle(stderr or "", max_chars=max_output_chars),
            duration_sec=duration,
            timed_out=timed_out,
        )

