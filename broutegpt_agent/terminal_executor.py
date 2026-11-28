import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any


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
    sandboxed: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def format_for_llm(self, max_chars: int = 14000) -> str:
        status = "TIMEOUT" if self.timed_out else f"EXIT_CODE={self.exit_code}"
        sand_str = " [SANDBOXED]" if self.sandboxed else ""
        combined = f"$ {self.command}\n[{status}{sand_str} in {self.duration_sec:.2f}s]\n"
        if self.stdout.strip():
            combined += f"--- STDOUT ---\n{self.stdout.strip()}\n"
        if self.stderr.strip():
            combined += f"--- STDERR ---\n{self.stderr.strip()}\n"
        return _truncate_middle(combined, max_chars=max_chars)


SCRUBBED_ENV_EXACT = {"GITHUB_TOKEN"}
SCRUBBED_ENV_SUFFIXES = ("_API_KEY", "_TOKEN")
SCRUBBED_ENV_PREFIXES = ("AWS_",)


def _scrub_child_env(base_env: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of the environment stripped of sensitive API keys, tokens, and cloud secrets."""
    cleaned: Dict[str, str] = {}
    for k, v in base_env.items():
        ku = k.upper()
        if (
            ku in SCRUBBED_ENV_EXACT
            or ku.endswith(SCRUBBED_ENV_SUFFIXES)
            or ku.startswith(SCRUBBED_ENV_PREFIXES)
        ):
            continue
        cleaned[k] = v
    return cleaned


class TerminalExecutor:
    """
    Workspace-jailed Zsh/shell executor with process-group timeout termination,
    path/command boundary enforcement, credential scrubbing, macOS Seatbelt,
    and Linux bubblewrap sandboxing.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        default_timeout: int = 90,
        use_kernel_sandbox: bool = True,
        strict_sandbox: bool = False,
    ) -> None:
        self.workspace_root = Path(workspace_dir).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout
        self.use_kernel_sandbox = bool(use_kernel_sandbox)
        self.strict_sandbox = bool(strict_sandbox)

    def resolve_safe_path(self, target_path: str | Path) -> Path:
        raw = Path(target_path)
        candidate = (raw if raw.is_absolute() else (self.workspace_root / raw)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                f"Workspace jail violation: '{target_path}' resolves outside workspace root '{self.workspace_root}'"
            ) from exc
        return candidate

    def validate_command(self, command: str) -> None:
        if not command or not command.strip():
            raise ValueError("Empty command string rejected.")
        for pattern in BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, command):
                raise ValueError(f"Command blocked by safety filter pattern: '{pattern}'")

    def _sb_escape(self, p: str) -> str:
        return p.replace("\\", "\\\\").replace('"', '\\"')

    def is_sandbox_disabled(self) -> bool:
        return os.getenv("BROUTEGPT_NO_SANDBOX", "").strip().lower() in ("1", "true", "yes")

    def is_macos_sandbox_available(self) -> bool:
        if self.is_sandbox_disabled():
            return False
        return bool(
            self.use_kernel_sandbox
            and sys.platform == "darwin"
            and Path("/usr/bin/sandbox-exec").is_file()
        )

    def get_bubblewrap_path(self) -> Optional[str]:
        if self.is_sandbox_disabled() or not self.use_kernel_sandbox:
            return None
        if not sys.platform.startswith("linux"):
            return None
        return shutil.which("bwrap")

    def _build_seatbelt_profile(self) -> str:
        ws_path = str(self.workspace_root)
        home_path = str(Path.home().resolve())
        escaped_ws = self._sb_escape(ws_path)
        escaped_home = self._sb_escape(home_path)
        git_hooks = self._sb_escape(str(self.workspace_root / ".git" / "hooks"))
        git_config = self._sb_escape(str(self.workspace_root / ".git" / "config"))
        state_dir = self._sb_escape(str(self.workspace_root / ".broutegpt"))
        ssh_dir = self._sb_escape(str(Path(home_path) / ".ssh"))
        aws_dir = self._sb_escape(str(Path(home_path) / ".aws"))
        gnupg_dir = self._sb_escape(str(Path(home_path) / ".gnupg"))

        base_policy = "(allow default)"
        if self.strict_sandbox:
            base_policy = (
                "(allow default)\n"
                "(deny network*)"
            )

        return f"""(version 1)
{base_policy}
(deny file-write*)
(allow file-write*
  (subpath "{escaped_ws}")
  (subpath "/tmp")
  (subpath "/private/tmp")
  (subpath "/var/folders")
  (subpath "/private/var/folders")
  (subpath "{escaped_home}/Library/Caches/org.swift.swiftpm")
  (subpath "{escaped_home}/Library/org.swift.swiftpm")
  (subpath "{escaped_home}/.gradle")
  (literal "/dev/null")
  (literal "/dev/zero")
  (literal "/dev/tty")
  (literal "/dev/dtracehelper")
  (regex #"^/dev/fd/")
  (regex #"^/dev/tty")
)
(deny file-write*
  (subpath "{git_hooks}")
  (literal "{git_config}")
  (subpath "{state_dir}")
)
(deny file-read*
  (subpath "{ssh_dir}")
  (subpath "{aws_dir}")
  (subpath "{gnupg_dir}")
)
"""

    def _build_bubblewrap_cmd(self, bwrap_bin: str, shell_bin: str, command: str) -> list[str]:
        home_path = Path.home().resolve()
        cmd: list[str] = [
            bwrap_bin,
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(self.workspace_root), str(self.workspace_root),
            "--chdir", str(self.workspace_root),
            "--unshare-pid",
        ]
        if self.strict_sandbox:
            cmd.append("--unshare-net")

        for protected_dir in (
            self.workspace_root / ".git" / "hooks",
            self.workspace_root / ".broutegpt",
        ):
            if protected_dir.exists():
                cmd.extend(["--ro-bind", str(protected_dir), str(protected_dir)])

        git_cfg = self.workspace_root / ".git" / "config"
        if git_cfg.exists():
            cmd.extend(["--ro-bind", str(git_cfg), str(git_cfg)])

        for secret_dir in (home_path / ".ssh", home_path / ".aws", home_path / ".gnupg"):
            if secret_dir.exists():
                cmd.extend(["--tmpfs", str(secret_dir)])

        for cache_dir in (
            home_path / ".cache" / "org.swift.swiftpm",
            home_path / ".gradle",
        ):
            if cache_dir.exists():
                cmd.extend(["--bind", str(cache_dir), str(cache_dir)])

        cmd.extend([shell_bin, "-lc", command])
        return cmd

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        max_output_chars: int = 16000,
        env_extra: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        self.validate_command(command)
        eff_timeout = timeout if timeout is not None else self.default_timeout

        shell_bin = "/bin/zsh" if Path("/bin/zsh").exists() else (shutil.which("bash") or "/bin/sh")

        env = _scrub_child_env(os.environ.copy())
        env["PROJECT_WORKSPACE"] = str(self.workspace_root)
        if env_extra:
            env.update(_scrub_child_env(env_extra))

        sandboxed = False
        profile_file = None
        cmd_args = []

        if self.is_macos_sandbox_available():
            try:
                profile_content = self._build_seatbelt_profile()
                tf = tempfile.NamedTemporaryFile(mode="w", suffix=".sb", prefix="broutegpt_sb_", delete=False, dir="/private/tmp")
                tf.write(profile_content)
                tf.close()
                profile_file = tf.name
                env["TMPDIR"] = "/private/tmp"
                cmd_args = ["/usr/bin/sandbox-exec", "-f", profile_file, shell_bin, "-lc", command]
                sandboxed = True
            except Exception:
                cmd_args = [shell_bin, "-lc", command]
                sandboxed = False
        else:
            bwrap_bin = self.get_bubblewrap_path()
            if bwrap_bin:
                cmd_args = self._build_bubblewrap_cmd(bwrap_bin, shell_bin, command)
                sandboxed = True
            else:
                cmd_args = [shell_bin, "-lc", command]
                sandboxed = False

        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd_args,
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
        finally:
            if profile_file:
                try:
                    Path(profile_file).unlink()
                except Exception:
                    pass

        duration = round(time.perf_counter() - t0, 3)
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=_truncate_middle(stdout or "", max_chars=max_output_chars),
            stderr=_truncate_middle(stderr or "", max_chars=max_output_chars),
            duration_sec=duration,
            timed_out=timed_out,
            sandboxed=sandboxed,
        )
