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

    def grep_search(
        self,
        pattern: str,
        rel_path: str = ".",
        max_matches: int = 50,
        context_lines: int = 3,
        glob: Optional[str] = None,
        case_insensitive: bool = False,
    ) -> str:
        """Search workspace files for a regex with surrounding context lines and optional glob/case filters."""
        import fnmatch
        import shutil
        import subprocess

        try:
            target = self.executor.resolve_safe_path(rel_path)
        except Exception as exc:
            raise ValueError(f"Path outside workspace jail: {rel_path}") from exc
        if not target.exists():
            raise FileNotFoundError(f"Path not found in workspace: {rel_path}")

        ctx = max(0, min(int(context_lines), 10))
        limit = max(1, int(max_matches))
        flags = re.IGNORECASE if case_insensitive else 0

        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

        files = self._iter_workspace_files()
        if target.is_file():
            files = [target] if target in files else []
        else:
            files = [f for f in files if f.is_relative_to(target)]

        if glob:
            files = [
                f for f in files
                if fnmatch.fnmatch(str(f.relative_to(self.workspace_root)), glob)
                or fnmatch.fnmatch(f.name, glob)
            ]

        total_hits = 0
        blocks: List[str] = []
        used_rg = False

        rg = shutil.which("rg")
        if rg and not target.is_file():
            try:
                cmd = [
                    rg,
                    "--line-number",
                    "--no-heading",
                    "--color",
                    "never",
                    "-C",
                    str(ctx),
                ]
                if case_insensitive:
                    cmd.append("-i")
                if glob:
                    cmd.extend(["-g", glob])
                cmd.extend([pattern, str(target)])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode in (0, 1):
                    used_rg = True
                    cur_lines: List[str] = []
                    cur_file: Optional[str] = None
                    for raw_ln in result.stdout.splitlines():
                        if raw_ln == "--":
                            if cur_lines:
                                total_hits += 1
                                if len(blocks) < limit:
                                    blocks.append("\n".join(cur_lines))
                                cur_lines = []
                            continue
                        m_match = re.match(r"^(.+?):(\d+):(.*)$", raw_ln)
                        m_ctx = re.match(r"^(.+?)-(\d+)-(.*)$", raw_ln)
                        m = m_match or m_ctx
                        if not m:
                            continue
                        abs_p = Path(m.group(1))
                        try:
                            rel = str(abs_p.relative_to(self.workspace_root))
                        except ValueError:
                            continue
                        rel_parts = Path(rel).parts
                        if any(p in IGNORE_DIRS or p.startswith(".") for p in rel_parts):
                            continue
                        if cur_file is not None and rel != cur_file and cur_lines:
                            total_hits += 1
                            if len(blocks) < limit:
                                blocks.append("\n".join(cur_lines))
                            cur_lines = []
                        cur_file = rel
                        sep = ":" if m_match else "-"
                        cur_lines.append(f"{rel}:{m.group(2)}{sep} {m.group(3)}")
                    if cur_lines:
                        total_hits += 1
                        if len(blocks) < limit:
                            blocks.append("\n".join(cur_lines))
            except Exception:
                used_rg = False

        if not used_rg:
            for fpath in files:
                rel = str(fpath.relative_to(self.workspace_root))
                try:
                    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    continue
                hit_indices = [i for i, ln in enumerate(lines) if compiled.search(ln)]
                if not hit_indices:
                    continue
                for h_idx in hit_indices:
                    total_hits += 1
                    if len(blocks) >= limit:
                        continue
                    lo = max(0, h_idx - ctx)
                    hi = min(len(lines), h_idx + ctx + 1)
                    snippet = []
                    for pos in range(lo, hi):
                        mark = ":" if pos == h_idx else "-"
                        snippet.append(f"{rel}:{pos + 1}{mark} {lines[pos]}")
                    blocks.append("\n".join(snippet))

        if not blocks:
            return "No matches found"

        out = "\n--\n".join(blocks)
        if total_hits > len(blocks):
            out += f"\n\n[Showing {len(blocks)} of {total_hits} match block(s); narrow with 'path' or 'glob' to inspect more]"
        return out

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

