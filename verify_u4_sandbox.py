import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.terminal_executor import TerminalExecutor, _scrub_child_env
from broutegpt_agent.dp_orchestrator import DPOrchestrator


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="u4_verify_") as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ".git" / "hooks").mkdir(parents=True, exist_ok=True)
        (ws / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
        (ws / ".broutegpt").mkdir(parents=True, exist_ok=True)

        # 1. Credential scrubbing in _scrub_child_env and TerminalExecutor.run
        dirty_env = {
            "PATH": "/usr/bin:/bin",
            "GEMINI_API_KEY": "secret-gem",
            "GROQ_API_KEY": "secret-groq",
            "MY_CUSTOM_TOKEN": "secret-tok",
            "GITHUB_TOKEN": "ghp_secret",
            "AWS_ACCESS_KEY_ID": "AKIA_SECRET",
            "AWS_SECRET_ACCESS_KEY": "SECRET_AWS",
            "SAFE_VAR": "keep_me",
        }
        scrubbed = _scrub_child_env(dirty_env)
        for forbidden in ("GEMINI_API_KEY", "GROQ_API_KEY", "MY_CUSTOM_TOKEN", "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            assert forbidden not in scrubbed, f"{forbidden} should be scrubbed"
        assert scrubbed.get("SAFE_VAR") == "keep_me"

        with patch.dict(os.environ, dirty_env, clear=False):
            ex = TerminalExecutor(ws, use_kernel_sandbox=True, strict_sandbox=False)
            res = ex.run(
                'printf "%s|%s|%s|%s|%s" "${GEMINI_API_KEY:-MISSING}" "${GITHUB_TOKEN:-MISSING}" "${AWS_ACCESS_KEY_ID:-MISSING}" "${MY_CUSTOM_TOKEN:-MISSING}" "${SAFE_VAR:-MISSING}"'
            )
            assert res.ok, f"Expected command success, got stderr: {res.stderr}"
            assert res.stdout.strip() == "MISSING|MISSING|MISSING|MISSING|keep_me", f"Unexpected env output: {res.stdout!r}"

        # 2. Seatbelt profile assertions (macOS)
        profile = ex._build_seatbelt_profile()
        assert 'org.swift.swiftpm' in profile, "SwiftPM cache path must remain allowed"
        assert '.gradle' in profile, "Gradle cache path must remain allowed"
        assert '(deny file-write*' in profile
        assert str(ws / ".git" / "hooks") in profile
        assert str(ws / ".git" / "config") in profile
        assert str(ws / ".broutegpt") in profile
        assert ".ssh" in profile and ".aws" in profile and ".gnupg" in profile
        assert "(deny network*)" not in profile

        strict_ex = TerminalExecutor(ws, use_kernel_sandbox=True, strict_sandbox=True)
        strict_profile = strict_ex._build_seatbelt_profile()
        assert "(deny network*)" in strict_profile, "strict_sandbox=True must include (deny network*)"

        # 3. BROUTEGPT_NO_SANDBOX & bubblewrap command builder
        with patch.dict(os.environ, {"BROUTEGPT_NO_SANDBOX": "1"}, clear=False):
            assert ex.is_sandbox_disabled() is True
            assert ex.is_macos_sandbox_available() is False
            assert ex.get_bubblewrap_path() is None
            no_sb_res = ex.run("echo no_sandbox_ok")
            assert no_sb_res.ok and not no_sb_res.sandboxed

        bwrap_cmd = strict_ex._build_bubblewrap_cmd("/usr/bin/bwrap", "/bin/zsh", "pytest")
        assert bwrap_cmd[0] == "/usr/bin/bwrap"
        assert "--ro-bind" in bwrap_cmd
        assert "--unshare-net" in bwrap_cmd

        # 4. DPOrchestrator & CLI wiring for strict_sandbox
        orch = DPOrchestrator(ws, strict_sandbox=True)
        assert orch.executor.strict_sandbox is True

        help_proc = subprocess.run(
            [sys.executable, str(ROOT / "broutegpt.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--strict-sandbox" in help_proc.stdout, "--strict-sandbox flag missing from --help"

    print("VERIFY_U4_SANDBOX: PASS")


if __name__ == "__main__":
    main()

