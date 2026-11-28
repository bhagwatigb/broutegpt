import json
from pathlib import Path
from typing import Any, Optional

from broutegpt import DEFAULT_CACHE_FILE, FreeLLMRouter
from .checkpoint_manager import CheckpointManager
from .condenser import RollingCondenser
from .dp_orchestrator import DPOrchestrator
from .event_stream import EventStream
from .router_adapter import CloudFirstRouter


def _resolve_resume_goal(project_root: Path, explicit_goal: Optional[str]) -> str:
    if explicit_goal and explicit_goal.strip():
        return explicit_goal.strip()
    stream = EventStream(project_root)
    goal_ev = next((e for e in stream.events if e.kind == "goal"), None)
    if goal_ev is not None and goal_ev.content.strip():
        return goal_ev.content.splitlines()[0].strip()
    plan_file = project_root / ".broutegpt" / "dp_plan.json"
    if plan_file.exists():
        try:
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
            if plan.get("overview"):
                return str(plan["overview"])
        except Exception:
            pass
    raise ValueError(
        f"No goal found to resume in '{project_root}'. Pass `--build \"<goal>\" --resume`."
    )


def run_agent_cli(
    build_goal: Optional[str] = None,
    inputs: Optional[list[str]] = None,
    project_dir: str = "./broutegpt_project",
    max_dps: int = 6,
    max_steps: int = 14,
    resume: bool = False,
    show_timeline: bool = False,
    inspect_step_id: Optional[int] = None,
    rollback_step_id: Optional[int] = None,
    restore_context: bool = True,
    rollback_scope: Optional[str] = None,
    branch_step_id: Optional[int] = None,
    branch_name: Optional[str] = None,
    preset_option: Optional[str] = None,
    force_refresh: bool = False,
    env_path: str = ".env",
    cache_path: str = DEFAULT_CACHE_FILE,
    router_override: Optional[CloudFirstRouter] = None,
    strict_sandbox: bool = False,
    auto_guard: bool = False,
) -> dict[str, Any]:
    """
    Entrypoint for autonomous zero-question DP building and time-travel forensics.
    """
    project_root = Path(project_dir).expanduser().resolve()

    # 1. Forensics-only operations (do not require initializing LLM network calls)
    if show_timeline or inspect_step_id is not None or rollback_step_id is not None or branch_step_id is not None:
        if not (project_root / ".broutegpt").exists():
            raise FileNotFoundError(
                f"No broutegpt state found in '{project_root}'. Run `--build` first."
            )
        stream = EventStream(project_root)
        condenser = RollingCondenser(project_root / ".broutegpt")
        ckpt = CheckpointManager(project_root, auto_main_git=True)

        if show_timeline:
            timeline = ckpt.format_timeline()
            print(f"\n=== Forensic Timeline: {project_root} ===\n{timeline}")
            return {"action": "timeline", "project_dir": str(project_root), "timeline": timeline}

        if inspect_step_id is not None:
            report = ckpt.inspect_step(inspect_step_id, stream=stream)
            print(f"\n{report}")
            return {"action": "inspect_step", "step_id": inspect_step_id, "report": report}

        if branch_step_id is not None:
            br = ckpt.branch_from_step(
                branch_step_id,
                branch_name=branch_name,
                stream=stream,
            )
            print(
                f"\n[Branch Created] Forked Step #{br['source_step_id']} into '{br['branch_name']}' "
                f"at {br['branch_dir']} (SHA {br['source_commit'][:12]})"
            )
            return {"action": "branch", **br}

        if rollback_step_id is not None:
            eff_scope = rollback_scope or ("both" if restore_context else "code")
            rb = ckpt.rollback_to_step(
                rollback_step_id,
                stream=stream,
                condenser=condenser,
                restore_context=(eff_scope in ("both", "chat")),
                reason=f"CLI rollback ({eff_scope}) to step #{rollback_step_id}",
                scope=eff_scope,
            )
            print(
                f"\n[Rollback Complete ({rb['scope']})] Restored Step #{rb['restored_step_id']} "
                f"(DP {rb['restored_dp_index']}, SHA {rb['restored_commit'][:12]}) | "
                f"Discarded events: {rb['discarded_events']} | New checkpoint: Step #{rb['new_checkpoint_step_id']}"
            )
            return {"action": "rollback", **rb}

    # 2. Autonomous Build / Resume
    goal = _resolve_resume_goal(project_root, build_goal) if resume else (build_goal or "").strip()
    if not goal:
        raise ValueError("Provide a project goal via `--build \"...\"` or pass `--resume`.")

    if router_override is not None:
        cfr = router_override
    else:
        base_router = FreeLLMRouter(
            env_path=env_path,
            cache_path=cache_path,
            force_refresh=force_refresh,
        )
        if preset_option:
            base_router.configure_non_interactive(preset_option)
        cfr = CloudFirstRouter(router=base_router)

    print("\n============================================================")
    print("  BROUTEGPT — AUTONOMOUS ZERO-QUESTION DP BUILDER")
    print("============================================================")
    print(f"  Project Dir : {project_root}")
    print(f"  Goal        : {goal}")
    print(f"  Inputs      : {', '.join(inputs) if inputs else 'None'}")
    print(f"  Max DPs     : {max_dps} | Max Steps/DP: {max_steps}")
    print("  Routing     : Heavy Deep-Reasoning Architect & Build -> Fast Small Debugger -> Deep Escalation -> Last-Resort Ollama")
    print("============================================================\n")

    orch = DPOrchestrator(
        project_dir=project_root,
        router=cfr,
        max_steps_per_dp=max_steps,
        max_dps=max_dps,
        auto_main_git=True,
        strict_sandbox=strict_sandbox,
        auto_guard=auto_guard,
    )
    summary = orch.run(goal=goal, inputs=inputs, resume=resume)

    print("\n============================================================")
    print("  AUTONOMOUS BUILD COMPLETE")
    print("============================================================")
    print(f"  Project     : {summary['project_name']} ({summary['project_dir']})")
    print(f"  Entrypoint  : {summary['entrypoint'] or 'see repo files'}")
    print(f"  Verified DPs: {len(summary['results'])}/{summary['total_dps']} in {summary['duration_sec']}s")
    print(f"  Checkpoint  : Step #{summary['latest_checkpoint']} (run `--timeline --project-dir {project_root}` to inspect)")
    print(f"  Token Usage : {summary.get('total_tokens', 0)} total tokens "
          f"({summary.get('total_prompt_tokens', 0)} prompt + {summary.get('total_completion_tokens', 0)} completion) "
          f"across {summary.get('llm_calls', 0)} LLM call(s)")
    print("============================================================\n")
    return summary

