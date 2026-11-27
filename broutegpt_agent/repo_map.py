import ast
import re
from pathlib import Path
from typing import List, Optional

from .terminal_executor import TerminalExecutor


IGNORE_DIRS = {
    ".git",
    ".broutegpt",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
}


def _format_arg(arg: ast.arg) -> str:
    if arg.annotation is None:
        return arg.arg
    try:
        return f"{arg.arg}: {ast.unparse(arg.annotation)}"
    except Exception:
        return arg.arg


def _format_fn_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    parts: List[str] = []

    posonly = getattr(node.args, "posonlyargs", [])
    regular = node.args.args
    all_pos = list(posonly) + list(regular)
    defaults = [None] * (len(all_pos) - len(node.args.defaults)) + list(node.args.defaults)

    for idx, (arg, default) in enumerate(zip(all_pos, defaults)):
        item = _format_arg(arg)
        if default is not None:
            try:
                item += f"={ast.unparse(default)}"
            except Exception:
                item += "=..."
        parts.append(item)
        if posonly and idx == len(posonly) - 1:
            parts.append("/")

    if node.args.vararg is not None:
        parts.append(f"*{_format_arg(node.args.vararg)}")
    elif node.args.kwonlyargs:
        parts.append("*")

    for kw_arg, kw_def in zip(node.args.kwonlyargs, node.args.kw_defaults):
        item = _format_arg(kw_arg)
        if kw_def is not None:
            try:
                item += f"={ast.unparse(kw_def)}"
            except Exception:
                item += "=..."
        parts.append(item)

    if node.args.kwarg is not None:
        parts.append(f"**{_format_arg(node.args.kwarg)}")

    ret = ""
    if node.returns is not None:
        if isinstance(node.returns, ast.Name):
            ret = f" -> {node.returns.id}"
        elif isinstance(node.returns, ast.Constant):
            ret = f" -> {node.returns.value!r}"
        else:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret = ""

    return f"{prefix}{node.name}({', '.join(parts)}){ret}"


class RepoMapper:
    """
    Aider-inspired AST symbol map & numbered file reader.
    Compresses workspace architecture into class/function signatures with line
    numbers so models can locate targets without reading every file in full.
    """

    def __init__(self, executor: TerminalExecutor) -> None:
        self.executor = executor
        self.workspace_root = executor.workspace_root

    def _iter_workspace_files(self) -> List[Path]:
        files: List[Path] = []
        for path in sorted(self.workspace_root.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(self.workspace_root).parts
            if any(part in IGNORE_DIRS or part.startswith(".") or part.endswith(".egg-info") for part in rel_parts):
                continue
            if path.suffix.lower() in {".pyc", ".pyo", ".so", ".dylib", ".png", ".jpg", ".jpeg", ".pdf", ".zip"}:
                continue
            files.append(path)
        return files

    def _summarize_python_file(self, fpath: Path, rel_path: str) -> str:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        line_count = len(source.splitlines())
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            return f"{rel_path} ({line_count}L) [SyntaxError at L{exc.lineno}: {exc.msg}]"

        symbols: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    try:
                        bases.append(ast.unparse(b))
                    except Exception:
                        pass
                base_str = f"({', '.join(bases)})" if bases else ""
                symbols.append(f"  L{node.lineno}: class {node.name}{base_str}:")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(f"    L{item.lineno}: {_format_fn_signature(item)}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(f"  L{node.lineno}: {_format_fn_signature(node)}")

        header = f"{rel_path} ({line_count}L)"
        if not symbols:
            return header
        return header + "\n" + "\n".join(symbols)

    def _summarize_generic_file(self, fpath: Path, rel_path: str) -> str:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        ext = fpath.suffix.lower()
        header = f"{rel_path} ({len(lines)}L)"
        highlights: List[str] = []

        if ext in (".js", ".ts", ".jsx", ".tsx"):
            for idx, ln in enumerate(lines, 1):
                m = re.match(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const)\s+([A-Za-z0-9_$]+)", ln)
                if m:
                    highlights.append(f"  L{idx}: {ln.strip()[:90]}")
        elif ext == ".md":
            for idx, ln in enumerate(lines, 1):
                if ln.startswith(("# ", "## ", "### ")):
                    highlights.append(f"  L{idx}: {ln.strip()[:90]}")

        if not highlights:
            return header
        return header + "\n" + "\n".join(highlights[:20])

    def build_map(self, max_chars: int = 8000) -> str:
        files = self._iter_workspace_files()
        if not files:
            return "(Workspace is currently empty — no project files created yet.)"

        blocks: List[str] = []
        used_chars = 0
        for fpath in files:
            rel_path = str(fpath.relative_to(self.workspace_root))
            if fpath.suffix.lower() == ".py":
                block = self._summarize_python_file(fpath, rel_path)
            else:
                block = self._summarize_generic_file(fpath, rel_path)

            if used_chars + len(block) + 2 > max_chars:
                remaining = len(files) - len(blocks)
                blocks.append(f"... ({remaining} additional file(s) omitted to fit repo-map budget)")
                break
            blocks.append(block)
            used_chars += len(block) + 2

        return "\n".join(blocks)

    def read_file_numbered(
        self,
        rel_path: str,
        start_line: int = 1,
        end_line: int = 2000,
    ) -> str:
        target = self.executor.resolve_safe_path(rel_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {rel_path}")
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {rel_path}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start_line))
        end = min(total, max(start, int(end_line)))
        numbered = [f"{i:>4} | {lines[i - 1]}" for i in range(start, end + 1)]
        header = f"=== {rel_path} (lines {start}-{end} of {total}) ==="
        return header + "\n" + ("\n".join(numbered) if numbered else "(empty file)")

    def list_dir(self, rel_path: str = ".") -> str:
        target = self.executor.resolve_safe_path(rel_path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {rel_path}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {rel_path}")

        entries: List[str] = []
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name in IGNORE_DIRS or item.name.startswith("."):
                continue
            tag = "[DIR] " if item.is_dir() else f"[{item.stat().st_size:,}B]"
            entries.append(f"  {tag:<10} {item.name}")
        return f"Directory '{rel_path}':\n" + ("\n".join(entries) if entries else "  (empty)")

    def grep_search(self, pattern: str, rel_path: str = ".", max_matches: int = 50) -> str:
        """Search workspace files for a regex, returning 'relative/path.py:LINE: text' lines."""
        try:
            target = self.executor.resolve_safe_path(rel_path)
        except Exception as exc:
            raise ValueError(f"Path outside workspace jail: {rel_path}") from exc
        if not target.exists():
            raise FileNotFoundError(f"Path not found in workspace: {rel_path}")

        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

        files = self._iter_workspace_files()
        if target.is_file():
            files = [target] if target in files else []
        else:
            files = [f for f in files if f.is_relative_to(target)]

        matches: List[str] = []

        def _scan(fpath: Path, text: str) -> bool:
            rel = str(fpath.relative_to(self.workspace_root))
            for idx, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    matches.append(f"{rel}:{idx}: {line}")
                    if len(matches) >= max_matches:
                        return True
            return False

        import shutil
        import subprocess

        rg = shutil.which("rg")
        if rg:
            try:
                result = subprocess.run(
                    [rg, "--line-number", "--no-heading", "--color", "never", pattern, str(target)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        parts = line.split(":", 2)
                        if len(parts) == 3:
                            abs_path = Path(parts[0])
                            try:
                                rel = str(abs_path.relative_to(self.workspace_root))
                            except ValueError:
                                continue
                            rel_parts = abs_path.relative_to(self.workspace_root).parts
                            if any(p in IGNORE_DIRS or p.startswith(".") for p in rel_parts):
                                continue
                            matches.append(f"{rel}:{parts[1]}: {parts[2]}")
                        if len(matches) >= max_matches:
                            break
            except Exception:
                rg = None

        if not rg:
            for fpath in files:
                if len(matches) >= max_matches:
                    break
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if _scan(fpath, text):
                    break

        if not matches:
            return "No matches found"
        return "\n".join(matches)

    def glob_files(self, pattern: str, max_results: int = 100) -> str:
        """Return sorted workspace-relative paths matching pattern (skips ignored/hidden)."""
        results: List[str] = []
        try:
            for path in self.workspace_root.glob(pattern):
                if not path.is_file():
                    continue
                rel_parts = path.relative_to(self.workspace_root).parts
                if any(
                    part in IGNORE_DIRS or part.startswith(".") or part.endswith(".egg-info")
                    for part in rel_parts
                ):
                    continue
                results.append(str(path.relative_to(self.workspace_root)))
                if len(results) >= max_results:
                    break
        except Exception:
            pass
        results.sort()
        return "\n".join(results) if results else "No matches found"

