import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checkpoint_manager import CheckpointManager
from .code_editor import CodeEditor, parse_agent_action, validate_action_json
from .condenser import RollingCondenser
from .event_stream import EventStream
from .input_inspector import InputInspector
from .repo_map import RepoMapper
from .router_adapter import CloudFirstRouter
from .terminal_executor import TerminalExecutor


PLAN_SYSTEM_PROMPT = (
    "You are an expert principal software architect designing an autonomous, "
    "zero-question Decision-Point (DP) execution plan. Break the project into "
    "2 to 6 sequential, self-contained Decision Points where each DP produces "
    "working, testable code and ends with a standalone Python verification script "
    "(`verify_dp1.py`, `verify_dp2.py`, etc.). Output ONLY valid JSON."
)

STEP_SYSTEM_PROMPT = """You are an autonomous senior software engineer executing a project Decision Point (DP) inside a macOS terminal workspace with ZERO human interaction.

Respond with ONLY ONE valid JSON action object per turn (no prose outside JSON). Choose strictly from these schemas:
1. Write a new or complete file:
   {"action": "write_file", "path": "relative/path.py", "content": "full file text..."}
2. Patch an existing file using surgical SEARCH/REPLACE blocks (preferred for edits):
   {"action": "patch_file", "path": "relative/path.py", "edits": [{"search": "exact old lines", "replace": "new lines"}], "fallback_content": "optional full file if patch misses"}
3. Read numbered lines of an existing file:
   {"action": "read_file", "path": "relative/path.py", "start_line": 1, "end_line": 200}
4. List directory contents:
   {"action": "list_dir", "path": "."}
5. Run a Zsh command in the workspace root (tests, scripts, pip/syntax checks):
   {"action": "run_shell", "command": "python3 verify_dp1.py"}
6. Declare the current DP complete ONLY AFTER `verify_dp<k>.py` exists and passed with exit code 0:
   {"action": "dp_complete", "summary": "concise summary of what was built and verified"}

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


class DPOrchestrator:
    """
    Zero-question autonomous Decision-Point builder:
      - Stage 0: Stages external input files into `sample_inputs/`, inspects schemas,
        and generates `dp_plan.json` (or resumes an existing plan).
      - Per-DP Loop: Mini-blueprint -> Build -> Static Linter -> `verify_dp{k}.py`
        -> Autonomous Debug Loop with Cloud-to-Local Tier Escalation -> Regression
        Guard across all prior `verify_dp*.py` scripts -> Shadow & Main Git Commit.
    """

    def __init__(
        self,
        project_dir: str | Path,
        router: Optional[CloudFirstRouter] = None,
        max_steps_per_dp: int = 14,
        max_dps: int = 6,
        command_timeout: int = 120,
        auto_main_git: bool = True,
    ) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.project_root / ".broutegpt"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.state_dir / "dp_plan.json"

        self.router = router if router is not None else CloudFirstRouter()
        self.executor = TerminalExecutor(self.project_root, default_timeout=command_timeout)
        self.inspector = InputInspector(router=self.router.router)
        self.stream = EventStream(self.project_root)
        self.condenser = RollingCondenser(self.state_dir, keep_tail_events=8, condense_batch_min=4)
        self.checkpoints = CheckpointManager(self.project_root, auto_main_git=auto_main_git)
        self.mapper = RepoMapper(self.executor)
        self.editor = CodeEditor(self.executor)

        self.max_steps_per_dp = max_steps_per_dp
        self.max_dps = max_dps

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

        return self.inspector.inspect(staged_paths if staged_paths else inputs)

    def _generate_master_plan(self, goal: str, input_summary: str) -> Dict[str, Any]:
        plan_prompt = (
            f"PROJECT GOAL:\n{goal}\n\n"
            f"AVAILABLE INPUT FILES IN WORKSPACE:\n{input_summary}\n\n"
            f"Design a sequential Decision-Point (DP) plan (maximum {self.max_dps} DPs) to build, "
            "verify, and deliver the complete working suite with zero human questions.\n"
            "The final DP must verify end-to-end execution against the input files (or realistic test fixtures if none were supplied) "
            "and produce all requested output artifacts.\n\n"
            "Return ONLY valid JSON in this exact schema:\n"
            "{\n"
            '  "project_name": "short_snake_case_name",\n'
            '  "overview": "1-2 sentence summary",\n'
            '  "entrypoint": "main_script_or_cli.py",\n'
            '  "dps": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "title": "Short DP Title",\n'
            '      "description": "Concrete modules/functions/CLI features to build in this DP",\n'
            '      "deliverables": ["file1.py", "verify_dp1.py"],\n'
            '      "verification_criteria": "Exact assertions verify_dp1.py must test"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        raw, used_model = self.router.query_autonomous(
            prompt=plan_prompt,
            system_prompt=PLAN_SYSTEM_PROMPT,
            tier="deep",
            temperature=0.1,
            validator=_validate_plan_json,
        )
        plan = _extract_json_object(raw)
        dps = plan.get("dps", [])[: self.max_dps]
        for idx, dp in enumerate(dps, 1):
            dp["id"] = idx
            dp.setdefault("status", "pending")
        plan["dps"] = dps
        plan["planned_by"] = used_model
        self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        return plan

    def _select_escalation_tier(self, consecutive_failures: int) -> str:
        if consecutive_failures <= 1:
            return "smart"
        if consecutive_failures <= 3:
            return "deep"
        return "max_context"

    def _run_regression_suite(self, up_to_dp: int) -> tuple[bool, str]:
        """Runs verify_dp1.py .. verify_dp{up_to_dp}.py sequentially to guarantee zero regressions."""
        logs: List[str] = []
        for k in range(1, up_to_dp + 1):
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
    ) -> Dict[str, Any]:
        dp_id = int(dp["id"])
        title = str(dp.get("title", f"DP {dp_id}"))
        verify_script = f"verify_dp{dp_id}.py"

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

        for step in range(1, self.max_steps_per_dp + 1):
            tier = self._select_escalation_tier(consecutive_failures)
            budget = 14000 if tier == "smart" else 24000
            compiled = self.condenser.compile_history(
                self.stream,
                target_token_budget=budget,
                active_dp_index=dp_id,
                router=self.router,
            )

            repo_map = self.mapper.build_map(max_chars=6500)
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
                "Choose your next single JSON action (`write_file`, `patch_file`, `read_file`, `list_dir`, `run_shell`, or `dp_complete`)."
            )

            raw_reply, used_model = self.router.query_autonomous(
                prompt=turn_prompt,
                system_prompt=STEP_SYSTEM_PROMPT,
                tier=tier,
                temperature=0.15,
                history=compiled["history"],
                validator=validate_action_json,
            )
            last_model_used = used_model
            action_obj = parse_agent_action(raw_reply)
            act = action_obj["action"]

            act_ev = self.stream.append(
                "action",
                json.dumps(action_obj, ensure_ascii=False),
                dp_index=dp_id,
                model=used_model,
            )

            # Execute action
            if act == "write_file":
                rel_path = str(action_obj["path"])
                content = str(action_obj["content"])
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
                        obs_text = outcome.message
                    else:
                        consecutive_failures += 1
                        obs_text = f"ERROR: {outcome.message}\nFix this syntax/static error immediately."
                except Exception as exc:
                    consecutive_failures += 1
                    obs_text = f"ERROR writing {rel_path}: {exc}"

                self.stream.append("observation", obs_text, dp_index=dp_id)

            elif act == "patch_file":
                rel_path = str(action_obj["path"])
                edits = action_obj.get("edits")
                if not isinstance(edits, list):
                    edits = [{"search": str(action_obj.get("search", "")), "replace": str(action_obj.get("replace", ""))}]
                fb = action_obj.get("fallback_content")
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
                        obs_text = outcome.message
                    else:
                        consecutive_failures += 1
                        obs_text = f"ERROR: {outcome.message}\nFix this syntax/static error immediately."
                except Exception as exc:
                    consecutive_failures += 1
                    obs_text = f"ERROR patching {rel_path}: {exc}"

                self.stream.append("observation", obs_text, dp_index=dp_id)

            elif act == "read_file":
                rel_path = str(action_obj["path"])
                s_line = int(action_obj.get("start_line") or 1)
                e_line = int(action_obj.get("end_line") or 250)
                try:
                    obs_text = self.mapper.read_file_numbered(rel_path, start_line=s_line, end_line=e_line)
                except Exception as exc:
                    obs_text = f"ERROR reading {rel_path}: {exc}"
                self.stream.append("observation", obs_text, dp_index=dp_id)

            elif act == "list_dir":
                rel_path = str(action_obj.get("path") or ".")
                try:
                    obs_text = self.mapper.list_dir(rel_path)
                except Exception as exc:
                    obs_text = f"ERROR listing {rel_path}: {exc}"
                self.stream.append("observation", obs_text, dp_index=dp_id)

            elif act == "run_shell":
                cmd = str(action_obj["command"])
                try:
                    res = self.executor.run(cmd)
                    self.checkpoints.capture(
                        dp_index=dp_id,
                        event_id=act_ev.id,
                        label=f"shell:{cmd[:40]}",
                        model=used_model,
                    )
                    if res.ok:
                        consecutive_failures = 0
                        obs_text = f"SHELL PASSED:\n{res.format_for_llm()}"
                    else:
                        consecutive_failures += 1
                        obs_text = f"SHELL FAILED:\n{res.format_for_llm()}\nAnalyze the traceback and fix the root cause."
                except Exception as exc:
                    consecutive_failures += 1
                    obs_text = f"SHELL BLOCKED/ERROR: {exc}"

                self.stream.append("observation", obs_text, dp_index=dp_id)

            elif act == "dp_complete":
                summary = str(action_obj.get("summary") or f"DP {dp_id} complete")
                passed, gate_log = self._run_regression_suite(up_to_dp=dp_id)
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
                self.stream.append(
                    "observation",
                    f"DP_COMPLETE REJECTED — Verification/Regression Gate Failed:\n{gate_log}\n"
                    f"Debug and fix the failing check before calling `dp_complete` again.",
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

    def run(
        self,
        goal: str,
        inputs: Any = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        input_report = self.stage_inputs(inputs)
        input_summary = input_report["brief_summary"]

        if resume and self.plan_path.exists():
            plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
            self.stream.append("note", "Resumed existing dp_plan.json run.", dp_index=0)
        else:
            self.stream.append("goal", f"{goal}\n\nStaged Inputs:\n{input_summary}", dp_index=0)
            plan = self._generate_master_plan(goal, input_summary)
            self.stream.append(
                "plan",
                json.dumps(plan, indent=2, ensure_ascii=False),
                dp_index=0,
                model=plan.get("planned_by"),
            )

        completed_results: List[Dict[str, Any]] = []
        for dp in plan.get("dps", []):
            if resume and dp.get("status") == "verified":
                completed_results.append({
                    "dp_id": dp["id"],
                    "title": dp.get("title", ""),
                    "status": "skipped_already_verified",
                })
                continue

            res = self._execute_single_dp(dp, goal=goal, input_summary=input_summary)
            dp["status"] = "verified"
            self.plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
            completed_results.append(res)

        elapsed = round(time.perf_counter() - t0, 2)
        summary_report = {
            "project_dir": str(self.project_root),
            "project_name": plan.get("project_name", self.project_root.name),
            "entrypoint": plan.get("entrypoint", ""),
            "total_dps": len(plan.get("dps", [])),
            "results": completed_results,
            "duration_sec": elapsed,
            "latest_checkpoint": self.checkpoints.records[-1].step_id if self.checkpoints.records else 0,
        }
        (self.state_dir / "build_report.json").write_text(
            json.dumps(summary_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary_report

