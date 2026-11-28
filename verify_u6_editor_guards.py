import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt_agent.code_editor import CodeEditor
from broutegpt_agent.terminal_executor import TerminalExecutor


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="u6_verify_") as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        ex = TerminalExecutor(ws)
        editor = CodeEditor(ex)

        # 1. Pre-existing file cannot be patched before read_file / record_read
        target_py = ws / "app.py"
        target_py.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

        blocked = False
        try:
            editor.patch_file("app.py", [{"search": "hello", "replace": "world"}])
        except RuntimeError as exc:
            blocked = "Read-before-edit" in str(exc)
        assert blocked, "Expected patch_file to fail before file was read"

        # 2. Reading file allows patch_file
        digest = editor.record_read("app.py")
        assert digest and len(digest) == 64
        ok_patch = editor.patch_file("app.py", [{"search": "hello", "replace": "world"}])
        assert ok_patch.ok and "world" in target_py.read_text(encoding="utf-8")

        # 3. External disk modification triggers staleness guard
        target_py.write_text("def greet():\n    return 'external_edit'\n", encoding="utf-8")
        stale_blocked = False
        try:
            editor.patch_file("app.py", [{"search": "external_edit", "replace": "patched"}])
        except RuntimeError as exc:
            stale_blocked = "Stale file guard" in str(exc)
        assert stale_blocked, "Expected patch_file to reject externally modified file"

        # Refresh read hash and verify syntax-invalid write & patch do NOT remain on disk
        editor.record_read("app.py")
        bad_patch = editor.patch_file("app.py", [{"search": "return 'external_edit'", "replace": "def broken(:"}])
        assert not bad_patch.ok
        assert "external_edit" in target_py.read_text(encoding="utf-8"), "Invalid patch must be rolled back on disk"

        bad_new = editor.write_file("broken_new.py", "def bad(:\n    pass\n")
        assert not bad_new.ok
        assert not (ws / "broken_new.py").exists(), "Syntax-invalid new file must not remain on disk"

        # 4. Expanded static checks: .yaml, .toml, .ts
        ok_yaml = editor.write_file("config.yaml", "service:\n  port: 8080\n")
        assert ok_yaml.ok
        bad_yaml = editor.write_file("bad.yaml", "service:\n\tport: [unclosed\n")
        assert not bad_yaml.ok and not (ws / "bad.yaml").exists()

        ok_toml = editor.write_file("pyproject.toml", '[project]\nname = "demo"\nversion = "0.1.0"\n')
        assert ok_toml.ok
        bad_toml = editor.write_file("bad.toml", '[project\nname = "broken"\n')
        assert not bad_toml.ok and not (ws / "bad.toml").exists()

        ok_ts = editor.write_file("index.ts", "export function add(a: number, b: number): number {\n  return a + b;\n}\n")
        assert ok_ts.ok
        bad_ts = editor.write_file("bad.ts", "export function add(a: number {\n  return a;\n}\n")
        assert not bad_ts.ok and not (ws / "bad.ts").exists()

    print("VERIFY_U6_EDITOR_GUARDS: PASS")


if __name__ == "__main__":
    main()

