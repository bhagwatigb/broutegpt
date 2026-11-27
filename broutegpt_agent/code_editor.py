import ast
import json
import py_compile
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .terminal_executor import TerminalExecutor


VALID_ACTIONS = {
    "write_file",
    "patch_file",
    "read_file",
    "list_dir",
    "run_shell",
    "grep_search",
    "glob_files",
    "dp_complete",
}

MAX_BATCH_ACTIONS = 6


def normalize_action_batch(action_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "actions" in action_obj and isinstance(action_obj["actions"], list):
        raw_actions = list(action_obj["actions"])
        if len(raw_actions) > MAX_BATCH_ACTIONS:
            print(f"[WARNING] Action batch size {len(raw_actions)} exceeds MAX_BATCH_ACTIONS={MAX_BATCH_ACTIONS}. Truncating to {MAX_BATCH_ACTIONS}.")
            raw_actions = raw_actions[:MAX_BATCH_ACTIONS]
        return raw_actions
    return [action_obj]


@dataclass
class StaticCheckResult:
    ok: bool
    file_path: str
    error: str = ""


@dataclass
class EditOutcome:
    ok: bool
    file_path: str
    mode_used: str  # write_file | exact_patch | normalized_patch | fallback_write
    static_check: StaticCheckResult
    message: str


def _escape_raw_newlines_in_json_strings(raw: str) -> str:
    """
    Repairs JSON strings where an LLM emitted literal unescaped newlines or tabs
    inside double-quoted string values.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            continue
        if in_string and ch == "\t":
            out.append("\\t")
            continue
        out.append(ch)
    return "".join(out)


def _normalize_single_action_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    action = str(item.get("action") or item.get("tool") or "").strip().lower()
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid or missing 'action': '{action}'. Must be one of {sorted(VALID_ACTIONS)}."
        )
    normalized = dict(item)
    normalized["action"] = action
    return normalized


def parse_agent_action(raw_response: str) -> Dict[str, Any]:
    """
    Extracts and validates a single JSON action envelope OR a multi-action batch
    envelope (`{"thought": "...", "actions": [...]}`) from model output.
    Raises ValueError with a concise diagnostic if the response is malformed.
    """
    text = raw_response.strip()
    if not text:
        raise ValueError("Empty model response.")

    candidates: List[str] = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced if block.strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    parsed: Optional[Dict[str, Any]] = None
    last_err = "No JSON object found in output."
    for cand in candidates:
        for attempt in (cand, _escape_raw_newlines_in_json_strings(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    parsed = obj
                    break
            except Exception as exc:
                last_err = str(exc)
        if parsed is not None:
            break

    if parsed is None:
        raise ValueError(f"Could not parse valid JSON action object: {last_err}")

    if "actions" in parsed:
        raw_actions = parsed.get("actions")
        if not isinstance(raw_actions, list) or not (1 <= len(raw_actions) <= MAX_BATCH_ACTIONS):
            raise ValueError(
                f"'actions' must be a non-empty list of 1 to {MAX_BATCH_ACTIONS} action objects."
            )
        normalized_list: List[Dict[str, Any]] = []
        for idx, sub in enumerate(raw_actions):
            if not isinstance(sub, dict):
                raise ValueError(f"actions[{idx}] must be a JSON object.")
            normalized_list.append(_normalize_single_action_dict(sub))
        parsed["actions"] = normalized_list
        return parsed

    return _normalize_single_action_dict(parsed)


def _validate_single_action_fields(obj: Dict[str, Any]) -> Optional[str]:
    act = obj["action"]
    if act == "write_file":
        if not obj.get("path") or "content" not in obj:
            return "write_file requires 'path' and 'content'"
    elif act == "patch_file":
        if not obj.get("path"):
            return "patch_file requires 'path'"
        has_edits = isinstance(obj.get("edits"), list) and len(obj["edits"]) > 0
        has_flat = ("search" in obj and "replace" in obj)
        if not has_edits and not has_flat:
            return "patch_file requires 'edits' list (or 'search' and 'replace')"
    elif act == "read_file":
        if not obj.get("path"):
            return "read_file requires 'path'"
    elif act == "run_shell":
        if not str(obj.get("command") or "").strip():
            return "run_shell requires non-empty 'command'"
    elif act in ("grep_search", "glob_files"):
        if not str(obj.get("pattern") or "").strip():
            return f"{act} requires non-empty 'pattern'"
    elif act == "dp_complete":
        if not str(obj.get("summary") or "").strip():
            return "dp_complete requires 'summary'"
    return None


def validate_action_json(raw_response: str) -> Optional[str]:
    """Router-compatible validator callback: returns None if valid, else error string."""
    try:
        obj = parse_agent_action(raw_response)
        if "actions" in obj:
            for idx, sub in enumerate(obj["actions"]):
                sub_err = _validate_single_action_fields(sub)
                if sub_err:
                    return f"actions[{idx}]: {sub_err}"
            return None
        return _validate_single_action_fields(obj)
    except Exception as exc:
        return str(exc)


def _normalized_block_replace(source_text: str, search_block: str, replace_block: str) -> Optional[str]:
    """
    Fallback matcher when exact substring replacement misses due to trailing
    whitespace differences on lines. Requires a unique contiguous line match.
    """
    src_lines = source_text.splitlines(keepends=True)
    srch_lines = search_block.splitlines()
    if not srch_lines:
        return None

    norm_srch = [ln.rstrip() for ln in srch_lines]
    n = len(norm_srch)
    matches: List[int] = []

    for i in range(len(src_lines) - n + 1):
        window = [src_lines[i + j].rstrip("\r\n").rstrip() for j in range(n)]
        if window == norm_srch:
            matches.append(i)

    if len(matches) != 1:
        return None

    idx = matches[0]
    newline = "\r\n" if (src_lines and src_lines[0].endswith("\r\n")) else "\n"
    rep_lines = [ln + newline for ln in replace_block.splitlines()]
    if replace_block == "":
        rep_lines = []
    elif not replace_block.endswith(("\n", "\r\n")) and idx + n == len(src_lines) and not src_lines[-1].endswith(("\n", "\r\n")):
        rep_lines[-1] = rep_lines[-1].rstrip("\r\n")

    merged = src_lines[:idx] + rep_lines + src_lines[idx + n :]
    return "".join(merged)


class CodeEditor:
    """
    Executes workspace-jailed file writes and surgical SEARCH/REPLACE patches,
    immediately running static syntax/fatal-lint verification on every touched file.
    """

    def __init__(self, executor: TerminalExecutor) -> None:
        self.executor = executor

    def static_check_file(self, rel_path: str) -> StaticCheckResult:
        target = self.executor.resolve_safe_path(rel_path)
        if not target.exists():
            return StaticCheckResult(ok=False, file_path=rel_path, error="File does not exist after write.")

        ext = target.suffix.lower()
        if ext == ".py":
            source = target.read_text(encoding="utf-8", errors="replace")
            try:
                ast.parse(source, filename=rel_path)
            except SyntaxError as exc:
                err_line = (exc.text or "").strip()
                return StaticCheckResult(
                    ok=False,
                    file_path=rel_path,
                    error=f"Python SyntaxError in {rel_path} at line {exc.lineno}, col {exc.offset}: {exc.msg} -> `{err_line}`",
                )
            try:
                py_compile.compile(str(target), doraise=True)
            except py_compile.PyCompileError as exc:
                return StaticCheckResult(ok=False, file_path=rel_path, error=f"py_compile failed: {exc.msg}")

            ruff_bin = shutil.which("ruff")
            if ruff_bin:
                res = subprocess.run(
                    [ruff_bin, "check", "--select", "E9,F63,F7,F82", str(target)],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    return StaticCheckResult(
                        ok=False,
                        file_path=rel_path,
                        error=f"Ruff fatal static check failed in {rel_path}:\n{(res.stdout or res.stderr).strip()}",
                    )
            return StaticCheckResult(ok=True, file_path=rel_path)

        if ext == ".json":
            try:
                json.loads(target.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                return StaticCheckResult(ok=False, file_path=rel_path, error=f"Invalid JSON in {rel_path}: {exc}")
            return StaticCheckResult(ok=True, file_path=rel_path)

        if ext in (".js", ".mjs"):
            node_bin = shutil.which("node")
            if node_bin:
                res = subprocess.run([node_bin, "--check", str(target)], capture_output=True, text=True)
                if res.returncode != 0:
                    return StaticCheckResult(
                        ok=False,
                        file_path=rel_path,
                        error=f"Node syntax check failed in {rel_path}:\n{(res.stderr or res.stdout).strip()}",
                    )
            return StaticCheckResult(ok=True, file_path=rel_path)

        if ext in (".sh", ".zsh"):
            res = subprocess.run(["/bin/zsh", "-n", str(target)], capture_output=True, text=True)
            if res.returncode != 0:
                return StaticCheckResult(
                    ok=False,
                    file_path=rel_path,
                    error=f"Shell syntax error in {rel_path}:\n{(res.stderr or res.stdout).strip()}",
                )
            return StaticCheckResult(ok=True, file_path=rel_path)

        return StaticCheckResult(ok=True, file_path=rel_path)

    def write_file(self, rel_path: str, content: str) -> EditOutcome:
        target = self.executor.resolve_safe_path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        final_text = content if (not content or content.endswith("\n")) else (content + "\n")
        target.write_text(final_text, encoding="utf-8")

        check = self.static_check_file(rel_path)
        if not check.ok:
            return EditOutcome(
                ok=False,
                file_path=rel_path,
                mode_used="write_file",
                static_check=check,
                message=f"Wrote {rel_path}, but STATIC CHECK FAILED:\n{check.error}",
            )
        return EditOutcome(
            ok=True,
            file_path=rel_path,
            mode_used="write_file",
            static_check=check,
            message=f"Wrote {rel_path} ({len(final_text.splitlines())} lines) — static check PASSED.",
        )

    def patch_file(
        self,
        rel_path: str,
        edits: List[Dict[str, str]],
        fallback_content: Optional[str] = None,
    ) -> EditOutcome:
        target = self.executor.resolve_safe_path(rel_path)
        if not target.exists():
            if fallback_content is not None:
                return self.write_file(rel_path, fallback_content)
            raise FileNotFoundError(f"Cannot patch missing file: {rel_path}")

        current = target.read_text(encoding="utf-8", errors="replace")
        mode_used = "exact_patch"

        try:
            for idx, edit in enumerate(edits, 1):
                search = str(edit.get("search", ""))
                replace = str(edit.get("replace", ""))
                if not search:
                    raise ValueError(f"Edit #{idx} has empty 'search' block.")

                hits = current.count(search)
                if hits == 1:
                    current = current.replace(search, replace, 1)
                    continue
                if hits > 1:
                    raise ValueError(
                        f"Edit #{idx} search block matched {hits} locations in {rel_path}. Include 3-5 surrounding context lines."
                    )

                norm_updated = _normalized_block_replace(current, search, replace)
                if norm_updated is not None:
                    current = norm_updated
                    mode_used = "normalized_patch"
                    continue

                raise ValueError(
                    f"Edit #{idx} search block not found in {rel_path}. Read the file with read_file to inspect exact lines."
                )
        except Exception as patch_err:
            if fallback_content is not None:
                outcome = self.write_file(rel_path, fallback_content)
                outcome.mode_used = "fallback_write"
                outcome.message = f"Search/replace missed ({patch_err}); applied fallback_content -> {outcome.message}"
                return outcome
            raise

        final_text = current if (not current or current.endswith("\n")) else (current + "\n")
        target.write_text(final_text, encoding="utf-8")

        check = self.static_check_file(rel_path)
        if not check.ok:
            return EditOutcome(
                ok=False,
                file_path=rel_path,
                mode_used=mode_used,
                static_check=check,
                message=f"Patched {rel_path} ({mode_used}), but STATIC CHECK FAILED:\n{check.error}",
            )
        return EditOutcome(
            ok=True,
            file_path=rel_path,
            mode_used=mode_used,
            static_check=check,
            message=f"Patched {rel_path} ({len(edits)} block(s), {mode_used}) — static check PASSED.",
        )

