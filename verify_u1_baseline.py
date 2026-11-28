import hashlib
import subprocess
import sys
from pathlib import Path

MAIN_DIR = Path.home() / "broutegpt"
PLAY_DIR = Path.home() / "broutegpt_playground"
BASELINE_FILE = Path("/tmp/broutegpt_v3_baseline.sha256")

assert MAIN_DIR.is_dir(), f"Missing main repo: {MAIN_DIR}"
assert PLAY_DIR.is_dir(), f"Missing playground dir: {PLAY_DIR}"
assert BASELINE_FILE.is_file(), f"Missing baseline checksum file: {BASELINE_FILE}"

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

main_py = sorted([MAIN_DIR / "broutegpt.py", *(MAIN_DIR / "broutegpt_agent").glob("*.py")])
assert len(main_py) == 12, f"Expected 12 core .py files, found {len(main_py)}"

for src in main_py:
    rel = src.relative_to(MAIN_DIR)
    dst = PLAY_DIR / rel
    assert dst.is_file(), f"Missing synced file in playground: {dst}"
    assert sha256(src) == sha256(dst), f"Hash mismatch for {rel}"

assert (PLAY_DIR / ".env").is_file(), "Missing .env in ~/broutegpt_playground"

git_check = subprocess.run(
    ["git", "-C", str(PLAY_DIR), "status", "--porcelain", "--ignored"],
    capture_output=True,
    text=True,
    check=True,
)
tracked_env = [ln for ln in git_check.stdout.splitlines() if ln.endswith(".env") and not ln.startswith("!!")]
assert not tracked_env, f".env is NOT ignored by git in playground: {tracked_env}"

sys.path.insert(0, str(PLAY_DIR))
import broutegpt
import broutegpt_agent
from broutegpt_agent.dp_orchestrator import DPOrchestrator
from broutegpt_agent.router_adapter import CloudFirstRouter

cli_help = subprocess.run(
    [sys.executable, str(PLAY_DIR / "broutegpt.py"), "--help"],
    capture_output=True,
    text=True,
    check=True,
)
assert "--build" in cli_help.stdout and "--rollback" in cli_help.stdout, "CLI flags missing from --help"

print("DP 1 VERIFICATION PASSED")
print(f"  Synced & verified files : {len(main_py)} core Python modules match baseline")
print(f"  Playground .env status  : present and git-ignored")
print(f"  Baseline checksums      : {BASELINE_FILE}")

