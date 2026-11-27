import json
import subprocess
from pathlib import Path

project_dir = Path.home() / "broutegpt" / "workspace" / "tabdisplay"
print(f"=== Project Directory: {project_dir} (exists={project_dir.exists()}) ===")

if not project_dir.exists():
    raise SystemExit("Workspace directory not found.")

# 1. Check dp_plan.json
plan_file = project_dir / ".broutegpt" / "dp_plan.json"
if plan_file.exists():
    plan = json.loads(plan_file.read_text())
    print("\n=== Decision-Point Plan Status ===")
    for dp in plan.get("decision_points", plan.get("dps", [])):
        idx = dp.get("index") or dp.get("dp_index")
        title = dp.get("title") or dp.get("name")
        status = dp.get("status", "unknown")
        print(f"  DP {idx}: {title} -> [{status}]")
else:
    print("\nNo dp_plan.json found.")

# 2. Check Git tags and latest checkpoints
tags = subprocess.run(["git", "tag"], cwd=project_dir, capture_output=True, text=True)
print(f"\n=== Git Verified Tags ===\n  {tags.stdout.strip() or 'None'}")

ckpt_file = project_dir / ".broutegpt" / "checkpoints.json"
if ckpt_file.exists():
    ckpts = json.loads(ckpt_file.read_text())
    entries = ckpts if isinstance(ckpts, list) else ckpts.get("checkpoints", [])
    print(f"Total Shadow-Git Checkpoints: {len(entries)}")
    if entries:
        last = entries[-1]
        print(f"Latest Checkpoint: #{last.get('step_id', last.get('id'))} | DP {last.get('dp_index')} | {last.get('summary', last.get('action', ''))}")

# 3. Check ADB device connection
adb = subprocess.run(["adb", "devices"], capture_output=True, text=True)
print(f"\n=== ADB Devices ===\n{adb.stdout.strip()}")

# 4. List all project files (excluding build artifacts and .git)
print("\n=== Existing Project Files ===")
for p in sorted(project_dir.rglob("*")):
    if p.is_file() and not any(part in {".git", ".broutegpt", ".build", "build", ".gradle"} for part in p.parts):
        rel = p.relative_to(project_dir)
        lines = len(p.read_text(errors="ignore").splitlines())
        print(f"  {rel} ({lines} lines)")

# 5. Print Swift files to verify real APIs vs stubs
swift_targets = [
    "mac_host/Sources/CVirtualDisplay/include/CGVirtualDisplayPrivate.h",
    "mac_host/Sources/CVirtualDisplay/shim.m",
    "mac_host/Sources/TabDisplayHost/VirtualDisplayManager.swift",
    "mac_host/Sources/TabDisplayHost/ScreenCaptureEncoder.swift",
    "mac_host/Sources/TabDisplayHost/main.swift",
]
for rel_path in swift_targets:
    fp = project_dir / rel_path
    print(f"\n==================== {rel_path} ====================")
    print(fp.read_text() if fp.exists() else "<MISSING>")

