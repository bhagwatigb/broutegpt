import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checkpoint_manager import CheckpointManager
from .code_editor import CodeEditor, normalize_action_batch, parse_agent_action, validate_action_json
from .condenser import RollingCondenser
from .event_stream import EventStream
from .hook_engine import HookEngine
from .input_inspector import InputInspector
from .mcp_bridge import MCPRegistry
from .permissions import PermissionPolicy
from .repo_map import RepoMapper
from .router_adapter import CloudFirstRouter
from .subagent import ExploreSubagent
from .terminal_executor import CommandResult, TerminalExecutor



PLAN_SYSTEM_PROMPT = (
    "You are an expert principal software architect designing an autonomous, "
    "zero-question Decision-Point (DP) execution plan. Break the project into "
    "2 to 6 sequential, self-contained Decision Points where each DP produces "
    "working, testable code and ends with a standalone Python verification script "
    "(`verify_dp1.py`, `verify_dp2.py`, etc.). Output ONLY valid JSON."
)

STEP_SYSTEM_PROMPT = """You are an autonomous senior software engineer executing a project Decision Point (DP) inside a macOS terminal workspace with ZERO human interaction.

Respond with ONLY valid JSON per turn (no prose outside JSON) — either a single action object (`{"thought": "...", "action": "..."}`) OR a multi-action batch (`{"thought": "...", "actions": [ {...}, {...} ]}`, up to 6 sequential actions per turn, e.g., `write_file` module + `write_file` `verify_dp<k>.py` + `run_shell` to test it in one turn). Sub-actions execute sequentially and halt immediately if any sub-action fails or triggers `dp_complete`. Every response MUST include a concise `"thought"` field (1-2 plain sentences explaining what you found, what bug/task you are addressing, and what this step executes).

Choose strictly from these action schemas (used directly or inside `"actions": [...]`):
1. Write a new or complete file:
   {"thought": "Creating parser.py to load FASTA sequences and compute GC content.", "action": "write_file", "path": "relative/path.py", "content": "full file text..."}
2. Patch an existing file using surgical SEARCH/REPLACE blocks (preferred for debugging/edits):
   {"thought": "verify_dp1.py raised ZeroDivisionError in cli.py; removing the invalid division.", "action": "patch_file", "path": "relative/path.py", "edits": [{"search": "exact old lines", "replace": "new lines"}], "fallback_content": "optional full file if patch misses"}
3. Read numbered lines of an existing file:
   {"thought": "Inspecting lines 1-80 of cli.py to locate the traceback line.", "action": "read_file", "path": "relative/path.py", "start_line": 1, "end_line": 200}
4. List directory contents:
   {"thought": "Checking generated files in the workspace.", "action": "list_dir", "path": "."}
5. Run a Zsh command in the workspace root (tests, scripts, pip/syntax checks):
   {"thought": "Running verify_dp1.py to verify parser outputs against sample_inputs.", "action": "run_shell", "command": "python3 verify_dp1.py"}
6. Declare the current DP complete ONLY AFTER `verify_dp<k>.py` exists and passed with exit code 0:
   {"thought": "verify_dp1.py passed all assertions with exit code 0; marking DP 1 complete.", "action": "dp_complete", "summary": "concise summary of what was built and verified"}
7. Search file contents with a regex pattern:
   {"thought": "Searching for 'def foo' across the repo to locate the function.", "action": "grep_search", "pattern": "def foo", "path": ".", "max_matches": 50}
8. Glob workspace files by pattern:
   {"thought": "Listing all Python files to plan edits.", "action": "glob_files", "pattern": "**/*.py", "max_results": 100}
9. Consult a deep-reasoning advisor model on demand without escalating the whole DP:
   {"thought": "Asking deep advisor to diagnose subtle edge case before patching.", "action": "consult_advisor", "question": "Why does parser fail on empty header?", "context": "optional snippet"}
10. Spawn an isolated read-only explore subagent so multi-file searches return only a concise summary:
   {"thought": "Exploring existing API routes across modules without filling context.", "action": "spawn_subagent", "task": "Find all CLI flags and config loaders", "path": "."}
11. Lazily load the full parameter schema of an MCP tool (or search available MCP tools) before calling it:
   {"thought": "Hydrating the schema for the MCP database tool before invoking it.", "action": "load_mcp_tool", "name": "server__tool_name"}
12. Invoke an external MCP tool after inspecting its schema:
   {"thought": "Calling the external MCP tool with validated arguments.", "action": "call_mcp_tool", "name": "server__tool_name", "arguments": {"key": "value"}}

Rules:
- Never ask questions or wait for confirmation.
- Keep all file paths relative to the workspace root.
- If sample input files were staged in `sample_inputs/`, ensure your code and `verify_dp<k>.py` test against those real input files and produce the exact requested outputs.
- Before calling `dp_complete`, you MUST create `verify_dp<k>.py` (using `assert` statements and exit code 0 on pass, non-zero on failure) and verify it via `run_shell`.
"""


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidates = [b.strip() for b in fenced if b.strip()]
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("Response did not contain a valid JSON object.")


def _validate_plan_json(raw_text: str) -> Optional[str]:
    try:
        obj = _extract_json_object(raw_text)
        dps = obj.get("dps")
        if not isinstance(dps, list) or not dps:
            return "Plan JSON must contain a non-empty 'dps' list."
        for idx, dp in enumerate(dps, 1):
            if not isinstance(dp, dict) or not dp.get("title"):
                return f"DP #{idx} must be an object with a 'title'."
        return None
    except Exception as exc:
        return str(exc)


MCP_ACTIONS = {"load_mcp_tool", "call_mcp_tool"}


def _validate_step_action(raw_text: str) -> Optional[str]:
    try:
        obj = _extract_json_object(raw_text)
    except Exception as exc:
        return str(exc)
    items = obj.get("actions") if isinstance(obj.get("actions"), list) else [obj]
    if not items:
        return "Action payload cannot be empty."
    if not any(isinstance(it, dict) and it.get("action") in MCP_ACTIONS for it in items):
        return validate_action_json(raw_text)
    for idx, it in enumerate(items[:6], 1):
        if not isinstance(it, dict) or not it.get("action"):
            return f"Action #{idx} must be an object with an 'action' field."
        act = str(it["action"])
        if act == "load_mcp_tool":
            if not str(it.get("name") or it.get("query") or "").strip():
                return f"Action #{idx} (load_mcp_tool) requires 'name' or 'query'."
        elif act == "call_mcp_tool":
            if not str(it.get("name") or "").strip():
                return f"Action #{idx} (call_mcp_tool) requires 'name'."
            args = it.get("arguments")
            if args is not None and not isinstance(args, dict):
                return f"Action #{idx} (call_mcp_tool) 'arguments' must be a JSON object."
        else:
            sub_err = validate_action_json(json.dumps(it, ensure_ascii=False))
            if sub_err:
                return sub_err
    return None


def _parse_step_action(raw_text: str) -> Dict[str, Any]:
    obj = _extract_json_object(raw_text)
    items = obj.get("actions") if isinstance(obj.get("actions"), list) else [obj]
    if any(isinstance(it, dict) and it.get("action") in MCP_ACTIONS for it in items):
        return obj
    return parse_agent_action(raw_text)


def _normalize_step_batch(action_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(action_obj.get("actions"), list):
        items = [it for it in action_obj["actions"][:6] if isinstance(it, dict)]
        if any(it.get("action") in MCP_ACTIONS for it in items):
            return items
    if action_obj.get("action") in MCP_ACTIONS:
        return [action_obj]
    return normalize_action_batch(action_obj)


def _format_console_snippet(text: str, max_lines: int = 8, max_chars: int = 650) -> str:
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "(no output)"
    if len(lines) > max_lines:
        kept = lines[:3] + [f"... ({len(lines) - 7} line(s) omitted) ..."] + lines[-4:]
    else:
        kept = lines
    joined = "\n    ".join(kept)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + " ..."
    return joined


class DPOrchestrator:
    """
    Zero-question autonomous Decision-Point builder:
      - Stage 0: Stages external input files into `sample_inputs/`, inspects schemas,
        and generates `dp_plan.json` (or resumes an existing plan).
      - Per-DP Loop: Mini-blueprint -> Build -> Static Linter -> `verify_dp{k}.py`
        -> Autonomous Debug Loop with Cloud-to-Local Tier Escalation -> Regression
        Guard across all prior `verify_dp*.py` scripts -> Shadow & Main Git Commit.
    """

    def read_memory(self) -> str:
        """Return persisted cross-session memory from .broutegpt/MEMORY.md."""
        return self.condenser.read_memory()

    def append_memory(self, entry: str, category: str = "note") -> str:
        """Persist a bounded cross-session note to .broutegpt/MEMORY.md."""
        if getattr(self, "bare", False):
            return ""
        return self.condenser.append_memory(entry, category=category)

    def _expand_rule_file(
        self,
        file_path: Path,
        visited: set[Path],
        depth: int = 0,
        max_depth: int = 5,
    ) -> str:
        resolved = file_path.expanduser().resolve()
        if resolved in visited or depth > max_depth or not resolved.is_file():
            return ""
        visited.add(resolved)
        try:
            raw_text = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

        out_lines: List[str] = []
        for line in raw_text.splitlines():
            m = re.match(r"^\s*@([^\s#]+)\s*$", line)
            if m:
                inc_raw = m.group(1).strip().strip("'\"")
                cand = Path(inc_raw).expanduser()
                if not cand.is_absolute():
                    rel_to_parent = (resolved.parent / cand).resolve()
                    rel_to_root = (self.project_root / cand).resolve()
                    cand = rel_to_parent if rel_to_parent.is_file() else rel_to_root
                else:
                    cand = cand.resolve()
                if cand.is_file() and cand not in visited:
                    imported = self._expand_rule_file(cand, visited=visited, depth=depth + 1, max_depth=max_depth)
                    if imported:
                        out_lines.append(imported)
                        continue
            out_lines.append(line)
        return "\n".join(out_lines)

    def _load_project_rules(self, include_memory: bool = True) -> str:
        """Load AGENTS.md, BROUTEGPT.md, CLAUDE.md (with recursive @path imports), and .broutegpt/MEMORY.md."""
        if getattr(self, "bare", False):
            return ""
        rule_paths = [
            self.project_root / "AGENTS.md",
            self.project_root / "BROUTEGPT.md",
            self.project_root / "CLAUDE.md",
            Path.home() / ".broutegpt" / "AGENTS.md",
            Path.home() / ".broutegpt" / "BROUTEGPT.md",
        ]
        visited: set[Path] = set()
        texts: List[str] = []
        for p in rule_paths:
            try:
                if p.is_file():
                    expanded = self._expand_rule_file(p, visited=visited)
                    if expanded.strip():
                        texts.append(expanded)
            except Exception:
                continue

        if include_memory:
            mem_text = self.read_memory().strip()
            if mem_text:
                texts.append(mem_text)

        return "\n\n".join(texts) if texts else ""



    def __init__(
        self,
        project_dir: str | Path,
        router: Optional[CloudFirstRouter] = None,
        max_steps_per_dp: int = 14,
        max_dps: int = 6,
        command_timeout: int = 120,
        auto_main_git: bool = True,
        strict_sandbox: bool = False,
        auto_guard: bool = False,
        max_parallel_workers: int = 4,
        bare: bool = False,
    ) -> None:

        self.project_root = Path(project_dir).expanduser().resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.project_root / ".broutegpt"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.state_dir / "dp_plan.json"

        self.hooks = HookEngine(self.project_root)
        self.router = router if router is not None else CloudFirstRouter(hook_engine=self.hooks)
        if getattr(self.router, "hook_engine", None) is None:
            self.router.hook_engine = self.hooks
        self.permissions = PermissionPolicy(
            self.project_root,
            router=self.router,
            auto_guard=auto_guard,
        )
        self.executor = TerminalExecutor(
            self.project_root,
            default_timeout=command_timeout,
            strict_sandbox=strict_sandbox,
        )

        self.inspector = InputInspector(router=self.router.router)
        self.stream = EventStream(self.project_root)
        self.condenser = RollingCondenser(
            self.state_dir,
            keep_tail_events=24,
            condense_batch_min=6,
            hook_engine=self.hooks,
        )
        self.checkpoints = CheckpointManager(self.project_root, auto_main_git=auto_main_git)
        self.mapper = RepoMapper(self.executor)
        self.editor = CodeEditor(self.executor)

        self.max_steps_per_dp = max_steps_per_dp
        self.max_dps = max_dps
        self.max_parallel_workers = max(1, int(max_parallel_workers))
        self.bare = bool(bare)
        self.mcp = MCPRegistry(self.project_root, enabled=(not self.bare))

    def stage_inputs(self, inputs: Any) -> Dict[str, Any]:
        """
        Copies user-supplied external files/folders into `<project_dir>/sample_inputs/`
        so workspace-jailed scripts and verification tests can read them directly.
        """
        if not inputs:
            return self.inspector.inspect(None)

        raw_items: List[str] = []
        if isinstance(inputs, (str, Path)):
            s = str(inputs).strip()
            if s:
                raw_items = [p.strip() for p in s.split(",")] if ("," in s and not Path(s).expanduser().exists()) else [s]
        elif isinstance(inputs, (list, tuple, set)):
            raw_items = [str(x).strip() for x in inputs if str(x).strip()]

        staged_paths: List[str] = []
        stage_dir = self.project_root / "sample_inputs"

        for item in raw_items:
            if item.startswith(("http://", "https://")):
                staged_paths.append(item)
                continue
            src = Path(item[1:] if item.startswith("@") else item).expanduser().resolve()
            if not src.exists():
                continue
            try:
                src.relative_to(self.project_root)
                staged_paths.append(str(src))
                continue
            except ValueError:
                pass

            stage_dir.mkdir(parents=True, exist_ok=True)
            dest = stage_dir / src.name
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            staged_paths.append(str(dest))

        report = self.inspector.inspect(staged_paths if staged_paths else inputs)
        if report["manifest"]:
            names = ", ".join(m["name"] for m in report["manifest"])
            print(f"[Stage 0 | Inputs] Staged {len(report['manifest'])} input(s) into sample_inputs/: {names}")
        return report

    def _generate_master_plan(self, goal: str, input_summary: str) -> Dict[str, Any]:
        print("[Stage 0 | Architect] Generating Decision-Point execution plan...")
        existing_repo_map = self.mapper.build_map(max_chars=6500)
        plan_prompt = (
            f"PROJECT GOAL:\n{goal}\n\n"
            f"EXISTING WORKSPACE CODEBASE MAP (use ONLY real existing classes/files shown here):\n{existing_repo_map}\n\n"
            f"AVAILABLE INPUT FILES IN WORKSPACE:\n{input_summary}\n\n"
            f"Design a detailed Decision-Point (DP) DAG execution plan (maximum {self.max_dps} DPs) to build, "
            "verify, and deliver the complete working suite with zero human questions.\n"
            "For each DP, specify exact module boundaries, function/CLI signatures, data structures, edge cases, concrete verification assertions, "
            "and optional `depends_on` prerequisite DP IDs (use `[]` for independent root DPs that can execute in parallel).\n"
            "The final DP must verify end-to-end execution against the input files (or realistic test fixtures if none were supplied) "
            "and produce all requested output artifacts.\n\n"
            "Return ONLY valid JSON in this exact schema:\n"
            "{\n"
            '  "project_name": "short_snake_case_name",\n'
            '  "overview": "2-3 sentence architectural summary",\n'
            '  "entrypoint": "main_script_or_cli.py",\n'
            '  "dps": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "title": "Short DP Title",\n'
            '      "depends_on": [],\n'
            '      "description": "Detailed modules, function signatures, CLI flags, and edge-case handling to build in this DP",\n'
            '      "deliverables": ["file1.py", "verify_dp1.py"],\n'
            '      "verification_criteria": "Exact edge-case and end-to-end assertions verify_dp1.py must test"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        raw, used_model = self.router.query_autonomous(
            prompt=plan_prompt,
            system_prompt=PLAN_SYSTEM_PROMPT,
            tier="deep",
            temperature=0.1,
            max_tokens=4096,
            validator=_validate_plan_json,
        )
        plan = _extract_json_object(raw)
        dps = plan.get("dps", [])[: self.max_dps]
        for idx, dp in enumerate(dps, 1):
            dp["id"] = idx
            dp.setdefault("status", "pending")
            if "depends_on" in dp and isinstance(dp["depends_on"], list):
                dp["depends_on"] = [int(x) for x in dp["depends_on"] if int(x) != idx]
        plan["dps"] = dps
        plan["planned_by"] = used_model
        self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[Plan Ready] {plan.get('project_name', 'project')} ({len(dps)} DP(s), entrypoint: {plan.get('entrypoint', 'N/A')}):")
        for dp in dps:
            deps = dp.get("depends_on")
            dep_str = f" (depends_on={deps})" if deps is not None else ""
            print(f"  - DP {dp['id']}: {dp.get('title')}{dep_str} -> deliverables: {dp.get('deliverables', [])}")
        print()
        return plan

    def compute_execution_waves(self, dps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups DPs into topological execution waves based on optional `depends_on` fields.
        - If no DP in `dps` specifies `depends_on`, falls back to sequential waves
          ([dp_1], [dp_2], ...) for full backward compatibility.
        - Raises ValueError on unknown dependency IDs, self-dependencies, or cycles.
        """
        if not dps:
            return []

        ordered = sorted(dps, key=lambda d: int(d["id"]))
        all_ids = {int(d["id"]) for d in ordered}
        has_explicit_dag = any(d.get("depends_on") is not None for d in ordered)

        deps_map: Dict[int, set[int]] = {}
        for idx, dp in enumerate(ordered):
            dp_id = int(dp["id"])
            if not has_explicit_dag:
                deps_map[dp_id] = {int(ordered[idx - 1]["id"])} if idx > 0 else set()
                continue

            raw_deps = dp.get("depends_on") or []
            if not isinstance(raw_deps, list):
                raise ValueError(f"DP {dp_id} 'depends_on' must be a list of DP IDs.")
            parsed_deps: set[int] = set()
            for dep in raw_deps:
                dep_id = int(dep)
                if dep_id == dp_id:
                    raise ValueError(f"DP {dp_id} cannot depend on itself.")
                if dep_id not in all_ids:
                    raise ValueError(f"DP {dp_id} depends on unknown DP ID {dep_id}.")
                parsed_deps.add(dep_id)
            deps_map[dp_id] = parsed_deps

        completed: set[int] = set()
        remaining: Dict[int, Dict[str, Any]] = {int(d["id"]): d for d in ordered}
        waves: List[List[Dict[str, Any]]] = []

        while remaining:
            ready_ids = sorted(
                dp_id
                for dp_id in remaining
                if deps_map[dp_id].issubset(completed)
            )
            if not ready_ids:
                unresolved = {k: sorted(deps_map[k] - completed) for k in sorted(remaining)}
                raise ValueError(f"Cycle detected in dp_plan.json depends_on DAG: {unresolved}")

            wave = [remaining.pop(dp_id) for dp_id in ready_ids]
            waves.append(wave)
            completed.update(ready_ids)

        return waves

    def _select_step_tier(
        self,
        step: int,
        consecutive_failures: int,
        has_written_code: bool,
    ) -> str:
        """
        Asymmetric role routing:
          - Heavy deep-reasoning models (`deep`) create the initial DP implementation
            and take back over if fast debugging fails >= 3 consecutive times.
          - Fast smaller models (`speed`) run verification scripts, inspect files,
            and apply quick surgical patches for minor bugs.
        """
        if consecutive_failures >= 4:
            return "max_context"
        if consecutive_failures >= 3:
            return "deep"
        if step == 1 or not has_written_code:
            return "deep"
        return "speed"

    def _run_regression_suite(
        self,
        up_to_dp: int,
        dp_ids: Optional[List[int]] = None,
    ) -> tuple[bool, str]:
        """Runs verify_dp<k>.py scripts sequentially to guarantee zero regressions."""
        target_ids = sorted(set(dp_ids)) if dp_ids is not None else list(range(1, up_to_dp + 1))
        logs: List[str] = []
        for k in target_ids:
            script_name = f"verify_dp{k}.py"
            script_path = self.project_root / script_name
            if not script_path.exists():
                return False, f"Missing required verification script `{script_name}`."
            res = self.executor.run(f"python3 {script_name}")
            logs.append(f"=== {script_name} (exit={res.exit_code}, {res.duration_sec}s) ===\n{res.format_for_llm(4000)}")
            if not res.ok:
                return False, "\n\n".join(logs)
        return True, "\n\n".join(logs)

    def _execute_single_dp(
        self,
        dp: Dict[str, Any],
        goal: str,
        input_summary: str,
        verify_dp_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        dp_id = int(dp["id"])
        title = str(dp.get("title", f"DP {dp_id}"))
        verify_script = f"verify_dp{dp_id}.py"

        print(f"--- Starting DP {dp_id}: {title} ---")
        dp_spec_text = (
            f"=== STARTING DECISION POINT {dp_id}: {title} ===\n"
            f"Description : {dp.get('description', '')}\n"
            f"Deliverables: {dp.get('deliverables', [])}\n"
            f"Verify Gate : {dp.get('verification_criteria', '')}\n"
            f"Required Verification Script: `{verify_script}`"
        )
        start_ev = self.stream.append("dp_start", dp_spec_text, dp_index=dp_id)
        dp_start_ckpt = self.checkpoints.capture(
            dp_index=dp_id,
            event_id=start_ev.id,
            label=f"dp_{dp_id}_start",
            model=None,
        )

        consecutive_failures = 0
        last_model_used = "system"
        did_auto_rollback = False
        has_written_code = False
        consecutive_reads = 0
        if self.stream.latest_by_kind("rules") is None:
            rules_text = self._load_project_rules()
            if rules_text:
                self.stream.append("rules", rules_text, dp_index=0)

        for step in range(1, self.max_steps_per_dp + 1):
            tier = self._select_step_tier(
                step=step,
                consecutive_failures=consecutive_failures,
                has_written_code=has_written_code,
            )
            fallback_budget = 32000 if tier == "speed" else 48000
            compiled = self.condenser.compile_history(
                self.stream,
                target_token_budget=fallback_budget,
                active_dp_index=dp_id,
                router=self.router,
                tier=tier,
            )

            repo_map = self.mapper.build_map(max_chars=6500)
            mcp_index = self.mcp.get_prompt_index()
            mcp_section = f"{mcp_index}\n\n" if mcp_index else ""
            turn_prompt = (
                f"ACTIVE DECISION POINT: DP {dp_id} — {title} (Step {step}/{self.max_steps_per_dp}, Escalation Tier: {tier})\n"
                f"GOAL: {goal}\n\n"
                f"DP {dp_id} REQUIREMENTS:\n"
                f"- Description: {dp.get('description', '')}\n"
                f"- Deliverables: {json.dumps(dp.get('deliverables', []))}\n"
                f"- Verification Criteria: {dp.get('verification_criteria', '')}\n"
                f"- Mandatory Verification Script: `{verify_script}`\n\n"
                f"INPUT FILES:\n{input_summary}\n\n"
                f"CURRENT WORKSPACE REPO MAP:\n{repo_map}\n\n"
                f"{mcp_section}"
                + "Choose your next single JSON action or action batch (`actions: [...]`) and include a concise `thought` explaining what you found and what you are changing or testing."
            )

            raw_reply, used_model = self.router.query_autonomous(
                prompt=turn_prompt,
                system_prompt=STEP_SYSTEM_PROMPT,
                tier=tier,
                temperature=0.15,
                history=compiled["history"],
                validator=_validate_step_action,
                use_native_tools=True,
            )

            last_model_used = used_model
            action_obj = _parse_step_action(raw_reply)
            sub_actions = _normalize_step_batch(action_obj)
            is_batch = len(sub_actions) > 1
            top_thought = str(action_obj.get("thought") or action_obj.get("diagnosis") or "").strip()

            print(f"[DP {dp_id} | Step {step}/{self.max_steps_per_dp}]" + (f" (batch of {len(sub_actions)} actions)" if is_batch else ""))
            if top_thought:
                print(f"  Finding/Plan : {top_thought}")

            batch_any_failed = False
            for batch_idx, sub_action in enumerate(sub_actions):
                act = sub_action["action"]
                sub_thought = str(sub_action.get("thought") or sub_action.get("diagnosis") or "").strip()
                if is_batch:
                    prefix_msg = f"  [Batch {batch_idx + 1}/{len(sub_actions)} | {act}]"
                    print(f"{prefix_msg} {sub_thought}" if sub_thought else prefix_msg)

                ev_meta = {"batch_index": batch_idx, "batch_size": len(sub_actions)} if is_batch else None
                act_ev = self.stream.append(
                    "action",
                    json.dumps(sub_action, ensure_ascii=False),
                    dp_index=dp_id,
                    model=used_model,
                    meta=ev_meta,
                )

                perm_res = self.permissions.evaluate(sub_action)
                if not perm_res.allowed:
                    consecutive_failures += 1
                    sub_failed = True
                    batch_any_failed = True
                    obs_text = f"PERMISSION DENIED ({perm_res.decision}): {perm_res.reason}"
                    if is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    print(f"  Perm Denied  : {act} -> {perm_res.reason}")
                    self.stream.append("observation", obs_text, dp_index=dp_id)
                    if is_batch and batch_idx + 1 < len(sub_actions):
                        print(f"  Batch Halted : Skipped remaining {len(sub_actions) - (batch_idx + 1)} sub-action(s) after failure.")
                    break

                pre_hook = self.hooks.dispatch(
                    "PreToolUse",
                    {
                        "tool": act,
                        "action": act,
                        "dp_id": dp_id,
                        "step": step,
                        "input": sub_action,
                    },
                )
                if pre_hook.blocked:
                    consecutive_failures += 1
                    sub_failed = True
                    batch_any_failed = True
                    obs_text = f"BLOCKED BY PreToolUse HOOK: {pre_hook.block_reason}"
                    if is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    print(f"  Hook Blocked : {act} -> {pre_hook.block_reason}")
                    self.stream.append("observation", obs_text, dp_index=dp_id)
                    if is_batch and batch_idx + 1 < len(sub_actions):
                        print(f"  Batch Halted : Skipped remaining {len(sub_actions) - (batch_idx + 1)} sub-action(s) after failure.")
                    break

                if pre_hook.additional_context:
                    self.stream.append(
                        "observation",
                        f"[PreToolUse additionalContext]\n{pre_hook.additional_context}",
                        dp_index=dp_id,
                    )

                sub_failed = False
                obs_text = ""

                # Execute sub-action and print live findings
                if act == "write_file":
                    rel_path = str(sub_action["path"])
                    content = str(sub_action["content"])
                    try:
                        outcome = self.editor.write_file(rel_path, content)
                        self.checkpoints.capture(
                            dp_index=dp_id,
                            event_id=act_ev.id,
                            label=f"write:{rel_path}",
                            model=used_model,
                        )
                        if outcome.ok:
                            consecutive_failures = 0
                            consecutive_reads = 0
                            has_written_code = True
                            obs_text = outcome.message
                            print(f"  Created/Wrote: {rel_path} ({len(content.splitlines())} lines) — syntax/static check PASSED")
                        else:
                            consecutive_failures += 1
                            sub_failed = True
                            obs_text = f"ERROR: {outcome.message}\nFix this syntax/static error immediately."
                            print(f"  Wrote File   : {rel_path} — STATIC CHECK FAILED:\n    {outcome.static_check.error}")
                    except Exception as exc:
                        consecutive_failures += 1
                        sub_failed = True
                        obs_text = f"ERROR writing {rel_path}: {exc}"
                        print(f"  Write Error  : {rel_path} -> {exc}")

                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "patch_file":
                    rel_path = str(sub_action["path"])
                    edits = sub_action.get("edits")
                    if not isinstance(edits, list):
                        edits = [{"search": str(sub_action.get("search", "")), "replace": str(sub_action.get("replace", ""))}]
                    fb = sub_action.get("fallback_content")
                    try:
                        outcome = self.editor.patch_file(
                            rel_path,
                            edits=edits,
                            fallback_content=str(fb) if fb is not None else None,
                        )
                        self.checkpoints.capture(
                            dp_index=dp_id,
                            event_id=act_ev.id,
                            label=f"patch:{rel_path}",
                            model=used_model,
                        )
                        if outcome.ok:
                            consecutive_failures = 0
                            consecutive_reads = 0
                            obs_text = outcome.message
                            print(f"  Debug/Patched: {rel_path} ({len(edits)} block(s), {outcome.mode_used}) — syntax/static check PASSED")
                        else:
                            consecutive_failures += 1
                            sub_failed = True
                            obs_text = f"ERROR: {outcome.message}\nFix this syntax/static error immediately."
                            print(f"  Patched File : {rel_path} — STATIC CHECK FAILED:\n    {outcome.static_check.error}")
                    except Exception as exc:
                        consecutive_failures += 1
                        sub_failed = True
                        obs_text = f"ERROR patching {rel_path}: {exc}"
                        print(f"  Patch Error  : {rel_path} -> {exc}")

                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "read_file":
                    rel_path = str(sub_action["path"])
                    s_line = int(sub_action.get("start_line") or 1)
                    e_line = int(sub_action.get("end_line") or 2000)
                    try:
                        obs_text = self.mapper.read_file_numbered(rel_path, start_line=s_line, end_line=e_line)
                        self.editor.record_read(rel_path)
                        print(f"  Inspected    : {rel_path} (lines {s_line}-{e_line})")
                        consecutive_reads += 1
                        if consecutive_reads >= 6:
                            sub_failed = True
                            consecutive_failures += 1
                            obs_text = "ERROR: Read-loop circuit breaker triggered after 6 consecutive read_file calls. You MUST execute patch_file, write_file, or run_shell now."
                        elif consecutive_reads >= 4:
                            obs_text += "\n\n[SYSTEM WARNING]: You have performed multiple consecutive reads. Please apply a patch_file or write_file to make progress."
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR reading {rel_path}: {exc}"
                        print(f"  Read Error   : {rel_path} -> {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "grep_search":
                    pattern = str(sub_action["pattern"])
                    rel_path = str(sub_action.get("path") or ".")
                    max_matches = int(sub_action.get("max_matches") or 50)
                    context_lines = int(sub_action.get("context_lines") if sub_action.get("context_lines") is not None else 3)
                    glob_filter = sub_action.get("glob")
                    case_insens = bool(sub_action.get("case_insensitive", False))
                    try:
                        obs_text = self.mapper.grep_search(
                            pattern,
                            rel_path,
                            max_matches=max_matches,
                            context_lines=context_lines,
                            glob=str(glob_filter) if glob_filter else None,
                            case_insensitive=case_insens,
                        )
                        print(f"  Grep Search : pattern='{pattern}' in {rel_path}")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR in grep_search: {exc}"
                        print(f"  Grep Error  : {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "glob_files":
                    pattern = str(sub_action["pattern"])
                    max_results = int(sub_action.get("max_results") or 100)
                    try:
                        obs_text = self.mapper.glob_files(pattern, max_results=max_results)
                        print(f"  Glob Files   : pattern='{pattern}'")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR in glob_files: {exc}"
                        print(f"  Glob Error   : {exc}")

                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "list_dir":
                    rel_path = str(sub_action.get("path") or ".")
                    try:
                        obs_text = self.mapper.list_dir(rel_path)
                        print(f"  Listed Dir   : {rel_path}")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR listing {rel_path}: {exc}"
                        print(f"  List Error   : {rel_path} -> {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "consult_advisor":
                    question = str(sub_action.get("question") or sub_action.get("query") or "").strip()
                    extra_ctx = str(sub_action.get("context") or "").strip()
                    advisor_prompt = (
                        f"ACTIVE DP {dp_id}: {title}\n"
                        f"GOAL: {goal}\n\n"
                        f"QUESTION FROM FAST WORKER:\n{question}\n"
                        + (f"\nRELEVANT CONTEXT:\n{extra_ctx}\n" if extra_ctx else "")
                        + "\nProvide a concise, actionable technical diagnosis and exact fix guidance."
                    )
                    try:
                        advice, advisor_model = self.router.query_autonomous(
                            prompt=advisor_prompt,
                            system_prompt="You are a senior principal software architect advising an autonomous coding agent. Be concise, concrete, and accurate.",
                            tier="deep",
                            temperature=0.1,
                            max_tokens=1500,
                        )
                        obs_text = f"ADVISOR RESPONSE ({advisor_model}):\n{advice.strip()}"
                        print(f"  Advisor      : Consulted {advisor_model} -> received {len(advice)} chars")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR consulting advisor: {exc}"
                        print(f"  Advisor Error: {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "run_shell":
                    cmd = str(sub_action["command"])
                    try:
                        res = self.executor.run(cmd)
                        self.checkpoints.capture(
                            dp_index=dp_id,
                            event_id=act_ev.id,
                            label=f"shell:{cmd[:40]}",
                            model=used_model,
                        )
                        raw_out = "\n".join(part for part in (res.stdout.strip(), res.stderr.strip()) if part)
                        snippet = _format_console_snippet(raw_out)
                        if res.ok:
                            consecutive_failures = 0
                            consecutive_reads = 0
                            obs_text = f"SHELL PASSED:\n{res.format_for_llm()}"
                            print(f"  Executed     : `{cmd}` -> PASSED (exit=0, {res.duration_sec}s)")
                            print(f"  Output       :\n    {snippet}")
                        else:
                            consecutive_failures += 1
                            sub_failed = True
                            obs_text = f"SHELL FAILED:\n{res.format_for_llm()}\nAnalyze the traceback and fix the root cause."
                            print(f"  Executed     : `{cmd}` -> FAILED (exit={res.exit_code}, {res.duration_sec}s)")
                            print(f"  Problem Found:\n    {snippet}")
                    except Exception as exc:
                        consecutive_failures += 1
                        sub_failed = True
                        obs_text = f"SHELL BLOCKED/ERROR: {exc}"
                        print(f"  Shell Error  : `{cmd}` -> {exc}")

                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "spawn_subagent":
                    sub_task = str(sub_action.get("task") or sub_action.get("query") or sub_action.get("objective") or "").strip()
                    scope_path = str(sub_action.get("path") or ".")
                    sub_turns = int(sub_action.get("max_turns") or 6)
                    try:
                        explorer = ExploreSubagent(
                            mapper=self.mapper,
                            router=self.router,
                            max_turns=sub_turns,
                        )
                        sub_rep = explorer.run(task=sub_task, scope_path=scope_path)
                        obs_text = (
                            f"SUBAGENT SUMMARY ({sub_rep.model_used}, {sub_rep.turns_used} turn(s)):\n"
                            f"{sub_rep.summary}"
                        )
                        print(f"  Subagent     : Explored '{sub_task[:50]}' in {sub_rep.turns_used} turn(s) via {sub_rep.model_used}")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR in spawn_subagent: {exc}"
                        print(f"  Subagent Err : {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "load_mcp_tool":
                    t_name = str(sub_action.get("name") or "").strip()
                    t_query = str(sub_action.get("query") or "").strip()
                    try:
                        hydrated = self.mcp.load_mcp_tool(name=t_name, query=t_query)
                        obs_text = f"MCP TOOL SCHEMA:\n{json.dumps(hydrated, indent=2, ensure_ascii=False)}"
                        print(f"  MCP Load     : '{t_name or t_query}' -> found={hydrated.get('found')}")
                    except Exception as exc:
                        sub_failed = True
                        obs_text = f"ERROR in load_mcp_tool: {exc}"
                        print(f"  MCP Load Err : {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "call_mcp_tool":
                    t_name = str(sub_action.get("name") or "").strip()
                    t_args = sub_action.get("arguments") if isinstance(sub_action.get("arguments"), dict) else {}
                    try:
                        mcp_out = self.mcp.call_mcp_tool(t_name, arguments=t_args)
                        obs_text = f"MCP TOOL RESULT ({t_name}):\n{mcp_out}"
                        print(f"  MCP Call     : {t_name} -> OK ({len(mcp_out)} chars)")
                    except Exception as exc:
                        consecutive_failures += 1
                        sub_failed = True
                        obs_text = f"ERROR in call_mcp_tool ({t_name}): {exc}"
                        print(f"  MCP Call Err : {t_name} -> {exc}")
                    if sub_failed and is_batch and batch_idx + 1 < len(sub_actions):
                        obs_text += f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed."
                    self.stream.append("observation", obs_text, dp_index=dp_id)

                elif act == "dp_complete":
                    summary = str(sub_action.get("summary") or f"DP {dp_id} complete")
                    passed, gate_log = self._run_regression_suite(up_to_dp=dp_id, dp_ids=verify_dp_ids)

                    if passed:
                        ver_ev = self.stream.append(
                            "dp_verified",
                            f"DP {dp_id} ({title}) VERIFIED.\nSummary: {summary}\n{gate_log}",
                            dp_index=dp_id,
                            model=used_model,
                        )
                        ckpt = self.checkpoints.commit_verified_dp(
                            dp_index=dp_id,
                            title=title,
                            event_id=ver_ev.id,
                            model=used_model,
                        )
                        self.append_memory(
                            f"DP {dp_id} ({title}) verified by {used_model}: {summary}",
                            category="dp_verified",
                        )
                        post_hook = self.hooks.dispatch(
                            "PostToolUse",
                            {
                                "tool": act,
                                "action": act,
                                "dp_id": dp_id,
                                "step": step,
                                "ok": True,
                                "output": summary,
                            },
                        )
                        if post_hook.additional_context:
                            self.stream.append(
                                "observation",
                                f"[PostToolUse additionalContext]\n{post_hook.additional_context}",
                                dp_index=dp_id,
                            )
                        if is_batch:
                            batch_hook = self.hooks.dispatch(
                                "PostToolBatch",
                                {
                                    "dp_id": dp_id,
                                    "step": step,
                                    "batch_size": len(sub_actions),
                                    "halted": False,
                                },
                            )
                            if batch_hook.additional_context:
                                self.stream.append(
                                    "observation",
                                    f"[PostToolBatch additionalContext]\n{batch_hook.additional_context}",
                                    dp_index=dp_id,
                                )
                        print(f"  Gate Check   : VERIFIED (`{verify_script}` + regression suite PASSED)")
                        print(f"  DP Summary   : {summary} [Checkpoint #{ckpt.step_id}]\n")
                        return {
                            "dp_id": dp_id,
                            "title": title,
                            "status": "verified",
                            "steps_taken": step,
                            "verified_by": used_model,
                            "checkpoint_step": ckpt.step_id,
                            "summary": summary,
                        }

                    consecutive_failures += 1
                    sub_failed = True
                    fail_hook = self.hooks.dispatch(
                        "DPVerifyFailed",
                        {
                            "dp_id": dp_id,
                            "title": title,
                            "step": step,
                            "gate_log": gate_log[:3000],
                        },
                    )
                    hook_extra = (
                        f"\n[DPVerifyFailed additionalContext]\n{fail_hook.additional_context}"
                        if fail_hook.additional_context
                        else ""
                    )
                    err_snippet = _format_console_snippet(gate_log)
                    print("  Gate Check   : REJECTED — verification script failed:")
                    print(f"    {err_snippet}")
                    halt_note = f"\n[BATCH HALTED]: Stopped after action {batch_idx + 1}/{len(sub_actions)} failed." if (is_batch and batch_idx + 1 < len(sub_actions)) else ""
                    obs_text = (
                        f"DP_COMPLETE REJECTED — Verification/Regression Gate Failed:\n{gate_log}\n"
                        f"Debug and fix the failing check before calling `dp_complete` again.{hook_extra}{halt_note}"
                    )
                    self.stream.append(
                        "observation",
                        obs_text,
                        dp_index=dp_id,
                    )

                post_hook = self.hooks.dispatch(
                    "PostToolUse",
                    {
                        "tool": act,
                        "action": act,
                        "dp_id": dp_id,
                        "step": step,
                        "ok": not sub_failed,
                        "output": obs_text[:2000],
                    },
                )
                if post_hook.additional_context:
                    self.stream.append(
                        "observation",
                        f"[PostToolUse additionalContext]\n{post_hook.additional_context}",
                        dp_index=dp_id,
                    )

                if sub_failed:
                    batch_any_failed = True
                    if is_batch and batch_idx + 1 < len(sub_actions):
                        print(f"  Batch Halted : Skipped remaining {len(sub_actions) - (batch_idx + 1)} sub-action(s) after failure.")
                    break

            if is_batch:
                batch_hook = self.hooks.dispatch(
                    "PostToolBatch",
                    {
                        "dp_id": dp_id,
                        "step": step,
                        "batch_size": len(sub_actions),
                        "halted": batch_any_failed,
                    },
                )
                if batch_hook.additional_context:
                    self.stream.append(
                        "observation",
                        f"[PostToolBatch additionalContext]\n{batch_hook.additional_context}",
                        dp_index=dp_id,
                    )


            # Auto-rollback files to DP start once if debug branch thrashes >= 4 consecutive failures
            if consecutive_failures >= 4 and not did_auto_rollback:
                did_auto_rollback = True
                self.checkpoints.rollback_to_step(
                    dp_start_ckpt.step_id,
                    stream=self.stream,
                    condenser=self.condenser,
                    restore_context=False,
                    reason=f"Auto-rollback workspace to DP {dp_id} start after {consecutive_failures} consecutive failures",
                )
                print(f"  Auto-Rollback: Restored workspace files to clean DP {dp_id} start (Step #{dp_start_ckpt.step_id}) while keeping error diagnostics.")
                self.stream.append(
                    "note",
                    f"Workspace files were rolled back to clean DP {dp_id} start (step #{dp_start_ckpt.step_id}) "
                    "while keeping error history so you can apply a clean fix.",
                    dp_index=dp_id,
                )

        raise RuntimeError(
            f"DP {dp_id} ('{title}') did not pass verification within {self.max_steps_per_dp} steps "
            f"(last model: {last_model_used}). Inspect timeline with `--timeline`."
        )

    def plan_only(
        self,
        goal: str,
        inputs: Any = None,
    ) -> Dict[str, Any]:
        """Stage 0 read-only planning: generates and saves dp_plan.json without executing any DP steps."""
        input_report = self.stage_inputs(inputs)
        input_summary = input_report["brief_summary"]
        self.stream.append("goal", f"{goal}\n\nStaged Inputs:\n{input_summary}", dp_index=0)
        rules_text = self._load_project_rules()
        if rules_text:
            self.stream.append("rules", rules_text, dp_index=0)
        plan = self._generate_master_plan(goal, input_summary)
        self.stream.append(
            "plan",
            json.dumps(plan, indent=2, ensure_ascii=False),
            dp_index=0,
            model=plan.get("planned_by"),
        )
        return plan

    def run(
        self,
        goal: str,
        inputs: Any = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sess_hook = self.hooks.dispatch(
            "SessionStart",
            {"goal": goal, "resume": resume, "project_dir": str(self.project_root)},
        )
        if sess_hook.blocked:
            raise RuntimeError(f"SessionStart blocked by hook: {sess_hook.block_reason}")
        if sess_hook.additional_context:
            self.stream.append(
                "observation",
                f"[SessionStart additionalContext]\n{sess_hook.additional_context}",
                dp_index=0,
            )

        input_report = self.stage_inputs(inputs)
        input_summary = input_report["brief_summary"]

        if resume and self.plan_path.exists():
            plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
            self.stream.append("note", "Resumed existing dp_plan.json run.", dp_index=0)
            rules_text = self._load_project_rules()
            if rules_text:
                self.stream.append("rules", rules_text, dp_index=0)
            print("[Resume] Loaded existing dp_plan.json")
        else:
            self.stream.append("goal", f"{goal}\n\nStaged Inputs:\n{input_summary}", dp_index=0)
            rules_text = self._load_project_rules()
            if rules_text:
                self.stream.append("rules", rules_text, dp_index=0)
            plan = self._generate_master_plan(goal, input_summary)
            self.stream.append(
                "plan",
                json.dumps(plan, indent=2, ensure_ascii=False),
                dp_index=0,
                model=plan.get("planned_by"),
            )

        waves = self.compute_execution_waves(plan.get("dps", []))
        verified_dp_ids: List[int] = [
            int(d["id"])
            for d in plan.get("dps", [])
            if resume and d.get("status") == "verified"
        ]
        completed_results: List[Dict[str, Any]] = []

        def _run_dp_in_worktree(
            target_dp: Dict[str, Any],
            wt_dir: str,
        ) -> Dict[str, Any]:
            wt_id = int(target_dp["id"])
            wt_orch = DPOrchestrator(
                project_dir=wt_dir,
                router=self.router,
                max_steps_per_dp=self.max_steps_per_dp,
                max_dps=self.max_dps,
                command_timeout=self.executor.default_timeout,
                auto_main_git=False,
                strict_sandbox=self.executor.strict_sandbox,
                auto_guard=self.permissions.auto_guard,
                max_parallel_workers=1,
            )
            wt_orch.stream.append("goal", f"{goal}\n\nStaged Inputs:\n{input_summary}", dp_index=0)
            prereq_ids = [
                int(dep)
                for dep in (target_dp.get("depends_on") or [])
                if (Path(wt_dir) / f"verify_dp{int(dep)}.py").exists()
            ]
            return wt_orch._execute_single_dp(
                target_dp,
                goal=goal,
                input_summary=input_summary,
                verify_dp_ids=sorted(set([*prereq_ids, wt_id])),
            )

        for wave_idx, wave in enumerate(waves, 1):
            pending_wave: List[Dict[str, Any]] = []
            for dp in wave:
                if resume and dp.get("status") == "verified":
                    completed_results.append({
                        "dp_id": dp["id"],
                        "title": dp.get("title", ""),
                        "status": "skipped_already_verified",
                    })
                else:
                    pending_wave.append(dp)

            if not pending_wave:
                continue

            if len(pending_wave) == 1:
                dp = pending_wave[0]
                dp_id = int(dp["id"])
                active_ids = sorted(set([*verified_dp_ids, dp_id]))
                res = self._execute_single_dp(
                    dp,
                    goal=goal,
                    input_summary=input_summary,
                    verify_dp_ids=active_ids,
                )
                dp["status"] = "verified"
                verified_dp_ids = active_ids
                self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
                completed_results.append(res)
                continue

            # Parallel wave: provision isolated git worktrees for all independent DPs
            wave_ids = [int(d["id"]) for d in pending_wave]
            print(f"[Wave {wave_idx} | Parallel DAG] Executing DPs {wave_ids} concurrently in isolated git worktrees...")
            pre_ev = self.stream.append(
                "note",
                f"Starting parallel DAG wave {wave_idx} for DPs {wave_ids}.",
                dp_index=0,
            )
            base_ckpt = self.checkpoints.capture(
                dp_index=0,
                event_id=pre_ev.id,
                label=f"wave_{wave_idx}_base",
                model=None,
            )

            wt_infos: Dict[int, Dict[str, Any]] = {}
            for dp in pending_wave:
                dp_id = int(dp["id"])
                wt_infos[dp_id] = self.checkpoints.create_dp_worktree(
                    dp_id=dp_id,
                    base_commit=base_ckpt.commit_hash,
                )

            branch_results: Dict[int, Dict[str, Any]] = {}
            branch_errors: Dict[int, BaseException] = {}
            workers = min(len(pending_wave), self.max_parallel_workers)
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fut_to_dp = {
                        pool.submit(_run_dp_in_worktree, dp, wt_infos[int(dp["id"])]["worktree_dir"]): dp
                        for dp in pending_wave
                    }
                    for fut in as_completed(fut_to_dp):
                        dp = fut_to_dp[fut]
                        dp_id = int(dp["id"])
                        try:
                            branch_results[dp_id] = fut.result()
                        except BaseException as exc:
                            branch_errors[dp_id] = exc
            finally:
                if branch_errors:
                    for dp_id, info in wt_infos.items():
                        self.checkpoints.cleanup_dp_worktree(dp_id, info["worktree_dir"])

            if branch_errors:
                first_id = sorted(branch_errors)[0]
                raise RuntimeError(
                    f"Parallel wave {wave_idx} failed in worktree DP {first_id}: {branch_errors[first_id]}"
                ) from branch_errors[first_id]

            # Merge verified worktrees cleanly into main workspace; fall back sequentially on conflicts
            conflicted_dps: List[Dict[str, Any]] = []
            merged_wave_dps: List[Dict[str, Any]] = []
            for dp in sorted(pending_wave, key=lambda d: int(d["id"])):
                dp_id = int(dp["id"])
                info = wt_infos[dp_id]
                wt_res = branch_results[dp_id]
                merge_ev = self.stream.append(
                    "note",
                    f"Merging worktree branch {info['branch_name']} for DP {dp_id}.",
                    dp_index=dp_id,
                    model=wt_res.get("verified_by"),
                )
                merge_out = self.checkpoints.merge_dp_worktree(
                    dp_id=dp_id,
                    worktree_dir=info["worktree_dir"],
                    title=str(dp.get("title", f"DP {dp_id}")),
                    event_id=merge_ev.id,
                    model=wt_res.get("verified_by"),
                )
                self.checkpoints.cleanup_dp_worktree(dp_id, info["worktree_dir"])

                if merge_out.get("merged"):
                    merged_wave_dps.append(dp)
                else:
                    print(
                        f"  [Merge Conflict] Worktree DP {dp_id} conflicted with an earlier branch; "
                        "falling back to sequential execution on merged workspace."
                    )
                    conflicted_dps.append(dp)

            if merged_wave_dps:
                merged_ids = sorted(set([*verified_dp_ids, *[int(d["id"]) for d in merged_wave_dps]]))
                reg_ok, reg_log = self._run_regression_suite(
                    up_to_dp=max(merged_ids),
                    dp_ids=merged_ids,
                )
                if not reg_ok:
                    raise RuntimeError(f"Post-merge regression suite failed after wave {wave_idx}:\n{reg_log}")

                for dp in merged_wave_dps:
                    dp_id = int(dp["id"])
                    wt_res = branch_results[dp_id]
                    ver_ev = self.stream.append(
                        "dp_verified",
                        f"DP {dp_id} ({dp.get('title', '')}) MERGED & VERIFIED.\nSummary: {wt_res.get('summary', '')}",
                        dp_index=dp_id,
                        model=wt_res.get("verified_by"),
                    )
                    ckpt = self.checkpoints.commit_verified_dp(
                        dp_index=dp_id,
                        title=str(dp.get("title", f"DP {dp_id}")),
                        event_id=ver_ev.id,
                        model=wt_res.get("verified_by"),
                    )
                    self.append_memory(
                        f"DP {dp_id} ({dp.get('title', '')}) verified in parallel worktree by {wt_res.get('verified_by')}: {wt_res.get('summary', '')}",
                        category="dp_verified",
                    )
                    dp["status"] = "verified"
                    wt_res["checkpoint_step"] = ckpt.step_id
                    wt_res["execution_mode"] = "parallel_worktree"
                    completed_results.append(wt_res)

                verified_dp_ids = merged_ids
                self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

            for dp in conflicted_dps:
                dp_id = int(dp["id"])
                active_ids = sorted(set([*verified_dp_ids, dp_id]))
                seq_res = self._execute_single_dp(
                    dp,
                    goal=goal,
                    input_summary=input_summary,
                    verify_dp_ids=active_ids,
                )
                seq_res["execution_mode"] = "sequential_conflict_fallback"
                dp["status"] = "verified"
                verified_dp_ids = active_ids
                self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
                completed_results.append(seq_res)

        elapsed = round(time.perf_counter() - t0, 2)

        # Aggregate per-DP token telemetry from the router call log.
        total_prompt_tokens = 0
        total_completion_tokens = 0
        llm_calls = 0
        for entry in getattr(self.router, "call_log", []) or []:
            if not isinstance(entry, dict) or entry.get("status") != "ok":
                continue
            llm_calls += 1
            total_prompt_tokens += int(entry.get("prompt_tokens") or 0)
            total_completion_tokens += int(entry.get("completion_tokens") or 0)
        total_tokens = total_prompt_tokens + total_completion_tokens

        summary_report = {
            "project_dir": str(self.project_root),
            "project_name": plan.get("project_name", self.project_root.name),
            "entrypoint": plan.get("entrypoint", ""),
            "total_dps": len(plan.get("dps", [])),
            "results": completed_results,
            "duration_sec": elapsed,
            "latest_checkpoint": self.checkpoints.records[-1].step_id if self.checkpoints.records else 0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        stop_hook = self.hooks.dispatch(
            "Stop",
            {
                "project_dir": str(self.project_root),
                "verified_dps": len(completed_results),
                "total_dps": len(plan.get("dps", [])),
            },
        )
        if stop_hook.additional_context:
            summary_report["stop_hook_context"] = stop_hook.additional_context
            self.stream.append(
                "observation",
                f"[Stop additionalContext]\n{stop_hook.additional_context}",
                dp_index=0,
            )

        (self.state_dir / "build_report.json").write_text(
            json.dumps(summary_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.mcp.close_all()
        return summary_report
