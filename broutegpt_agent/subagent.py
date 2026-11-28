import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .code_editor import _escape_raw_newlines_in_json_strings
from .condenser import RollingCondenser
from .event_stream import AgentEvent, EventStream
from .repo_map import RepoMapper
from .router_adapter import CloudFirstRouter


SUBAGENT_ALLOWED_ACTIONS = {
    "read_file",
    "grep_search",
    "glob_files",
    "list_dir",
    "subagent_done",
}

SUBAGENT_SYSTEM_PROMPT = """You are an isolated, strictly READ-ONLY repository exploration subagent.
Your job is to inspect code, search symbols/patterns, list directories, and return a concise, high-signal technical summary for the main coding agent.

Respond with ONLY a single valid JSON object per turn (no prose outside JSON).
Allowed actions ONLY:
1. Read numbered lines of a workspace file:
   {"thought": "Inspecting module signatures.", "action": "read_file", "path": "relative/path.py", "start_line": 1, "end_line": 250}
2. Regex search across files:
   {"thought": "Finding all callers of target function.", "action": "grep_search", "pattern": "def target", "path": ".", "max_matches": 40, "context_lines": 2}
3. Glob files by pattern:
   {"thought": "Listing test and config files.", "action": "glob_files", "pattern": "**/*.py", "max_results": 80}
4. List directory contents:
   {"thought": "Checking directory layout.", "action": "list_dir", "path": "."}
5. Finish exploration and return your synthesized findings:
   {"thought": "Gathered all requested file locations and signatures.", "action": "subagent_done", "summary": "Concise bullet-point findings with exact file paths and line numbers."}

Rules:
- You are strictly READ-ONLY. Never attempt write_file, patch_file, run_shell, or dp_complete.
- Keep your final `summary` factual, concise, and grounded in exact file paths and line numbers you inspected.
"""


def _parse_subagent_json(raw_response: str) -> Dict[str, Any]:
    text = raw_response.strip()
    if not text:
        raise ValueError("Empty subagent response.")

    candidates: List[str] = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidates.extend(b.strip() for b in fenced if b.strip())
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    parsed: Optional[Dict[str, Any]] = None
    last_err = "No JSON object found."
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
        raise ValueError(f"Invalid subagent JSON: {last_err}")

    act = str(parsed.get("action") or parsed.get("tool") or "").strip().lower()
    if act not in SUBAGENT_ALLOWED_ACTIONS:
        raise ValueError(
            f"Subagent action '{act}' is forbidden or invalid. Allowed read-only actions: {sorted(SUBAGENT_ALLOWED_ACTIONS)}"
        )
    if act == "read_file" and not parsed.get("path"):
        raise ValueError("read_file requires 'path'")
    if act in ("grep_search", "glob_files") and not str(parsed.get("pattern") or "").strip():
        raise ValueError(f"{act} requires non-empty 'pattern'")
    if act == "subagent_done" and not str(parsed.get("summary") or "").strip():
        raise ValueError("subagent_done requires non-empty 'summary'")

    out = dict(parsed)
    out["action"] = act
    return out


def _validate_subagent_json(raw_response: str) -> Optional[str]:
    try:
        _parse_subagent_json(raw_response)
        return None
    except Exception as exc:
        return str(exc)


class _IsolatedMemoryEventStream(EventStream):
    """
    In-memory EventStream subclass for explore subagents so intermediate reads/greps
    never touch or bloat `<project_root>/.broutegpt/events.jsonl`.
    """

    def __init__(self, project_root: Any) -> None:
        self.project_root = project_root
        self.state_dir = project_root / ".broutegpt"
        self.log_path = self.state_dir / "subagent_ephemeral.jsonl"
        self._events: List[AgentEvent] = []

    def _load(self) -> None:
        return None

    def append(
        self,
        kind: str,
        content: str,
        dp_index: int = 0,
        model: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        next_id = (self._events[-1].id + 1) if self._events else 1
        ev = AgentEvent(
            id=next_id,
            timestamp=0.0,
            dp_index=dp_index,
            kind=kind,
            content=content,
            model=model,
            meta=dict(meta or {}),
        )
        self._events.append(ev)
        return ev


@dataclass
class SubagentReport:
    ok: bool
    summary: str
    turns_used: int
    model_used: str
    events_recorded: int


class ExploreSubagent:
    """
    Executes bounded read-only codebase exploration in an isolated in-memory
    EventStream on a fast high-context model and returns only a compact summary.
    """

    def __init__(
        self,
        mapper: RepoMapper,
        router: CloudFirstRouter,
        max_turns: int = 6,
    ) -> None:
        self.mapper = mapper
        self.router = router
        self.max_turns = max(1, int(max_turns))

    def _select_fast_high_context_tier(self) -> str:
        # Prefer speed tier when active speed model has >= 32k context; otherwise smart/max_context
        limit_fn = getattr(self.router, "get_active_input_limit", None)
        if callable(limit_fn):
            try:
                speed_lim = limit_fn(tier="speed")
                if speed_lim is not None and int(speed_lim) >= 32000:
                    return "speed"
            except Exception:
                pass
        return "smart"

    def run(self, task: str, scope_path: str = ".") -> SubagentReport:
        iso_stream = _IsolatedMemoryEventStream(self.mapper.workspace_root)
        iso_condenser = RollingCondenser(
            self.mapper.workspace_root / ".broutegpt",
            keep_tail_events=12,
            condense_batch_min=999,
        )
        iso_stream.append(
            "goal",
            f"Explore Objective: {task}\nFocus Path: {scope_path}",
            dp_index=0,
        )

        tier = self._select_fast_high_context_tier()
        last_model = "system"
        collected_notes: List[str] = []

        for turn in range(1, self.max_turns + 1):
            compiled = iso_condenser.compile_history(
                iso_stream,
                target_token_budget=24000,
                active_dp_index=0,
                router=self.router,
                tier=tier,
            )
            repo_map = self.mapper.build_map(max_chars=5000)
            prompt = (
                f"EXPLORE TURN {turn}/{self.max_turns}\n"
                f"OBJECTIVE: {task}\n"
                f"FOCUS PATH: {scope_path}\n\n"
                f"WORKSPACE REPO MAP:\n{repo_map}\n\n"
                "Inspect the relevant files using read-only tools, or call `subagent_done` with a concise, exact summary."
            )

            raw_reply, used_model = self.router.query_autonomous(
                prompt=prompt,
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                tier=tier,
                temperature=0.1,
                max_tokens=1500,
                history=compiled["history"],
                validator=_validate_subagent_json,
            )
            last_model = used_model
            action_obj = _parse_subagent_json(raw_reply)
            act = action_obj["action"]
            iso_stream.append("action", json.dumps(action_obj, ensure_ascii=False), dp_index=0, model=used_model)

            if act == "subagent_done":
                summary = str(action_obj.get("summary") or "").strip()
                return SubagentReport(
                    ok=True,
                    summary=summary,
                    turns_used=turn,
                    model_used=used_model,
                    events_recorded=len(iso_stream.events),
                )

            try:
                if act == "read_file":
                    rel_path = str(action_obj["path"])
                    s_line = int(action_obj.get("start_line") or 1)
                    e_line = int(action_obj.get("end_line") or 400)
                    obs = self.mapper.read_file_numbered(rel_path, start_line=s_line, end_line=e_line)
                    collected_notes.append(f"Read {rel_path}:{s_line}-{e_line}")
                elif act == "grep_search":
                    obs = self.mapper.grep_search(
                        pattern=str(action_obj["pattern"]),
                        rel_path=str(action_obj.get("path") or scope_path or "."),
                        max_matches=int(action_obj.get("max_matches") or 40),
                        context_lines=int(action_obj.get("context_lines") if action_obj.get("context_lines") is not None else 2),
                        glob=str(action_obj["glob"]) if action_obj.get("glob") else None,
                        case_insensitive=bool(action_obj.get("case_insensitive", False)),
                    )
                    collected_notes.append(f"Searched '{action_obj['pattern']}'")
                elif act == "glob_files":
                    obs = self.mapper.glob_files(
                        pattern=str(action_obj["pattern"]),
                        max_results=int(action_obj.get("max_results") or 80),
                    )
                    collected_notes.append(f"Globbed '{action_obj['pattern']}'")
                elif act == "list_dir":
                    obs = self.mapper.list_dir(str(action_obj.get("path") or scope_path or "."))
                    collected_notes.append(f"Listed '{action_obj.get('path') or scope_path or '.'}'")
                else:
                    obs = f"ERROR: Unsupported read-only action '{act}'"
            except Exception as exc:
                obs = f"ERROR ({act}): {exc}"

            iso_stream.append("observation", obs[:8000], dp_index=0)

        fallback_summary = (
            f"Explore subagent reached max_turns={self.max_turns} on '{task}'. "
            f"Steps executed: {', '.join(collected_notes) if collected_notes else 'none'}."
        )
        return SubagentReport(
            ok=True,
            summary=fallback_summary,
            turns_used=self.max_turns,
            model_used=last_model,
            events_recorded=len(iso_stream.events),
        )

