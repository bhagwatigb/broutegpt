import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import DEFAULT_CACHE_FILE, _is_valid_key, _load_env_file
from .exporters import (
    SUPPORTED_OUTPUT_FORMATS,
    _build_format_system_instruction,
    _detect_output_format,
    _export_chatbot_output,
)
from .router import FreeLLMRouter
from .sources import _load_sources_payload


def _prompt_chatbot_sources(router: FreeLLMRouter) -> List[str]:
    print("\n=== Step 1: Add Source Material (Any Kind) ===")
    print("  Accepts: PDF, DOCX, PPTX, XLSX, MD/TXT/Code, Images (PNG/JPG/WEBP), Audio/Video (MP3/MP4/WAV), Folders, URLs, or 'paste'")
    print("  Tip: Separate multiple paths/URLs with commas, or press Enter for no sources.")
    raw = router._read_text_or_file_arg("  Sources: ", allow_dir=True, return_path_only=True)
    if not raw:
        return []
    if "," in raw and not Path(raw).expanduser().exists():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [raw]


def run_chatbot_cli(
    env_path: str = ".env",
    cache_path: str = DEFAULT_CACHE_FILE,
    force_refresh: bool = False,
    preset_sources: Optional[List[str]] = None,
    preset_option: Optional[str] = None,
    preset_format: str = "auto",
    output_dir: str = "chatbot_outputs",
    one_shot_prompt: Optional[str] = None,
    print_only: bool = False,
    bare: bool = False,
    cli_output_format: str = "text",
) -> Optional[Dict[str, Any]]:
    t_start = time.perf_counter()
    json_mode = str(cli_output_format).strip().lower() == "json"
    is_headless = bool(print_only or bare or (one_shot_prompt is not None) or json_mode)
    log_ctx = contextlib.redirect_stdout(sys.stderr) if (json_mode or print_only) else contextlib.nullcontext()

    with log_ctx:
        if not is_headless:
            print("\n============================================================")
            print("  FREE LLM ROUTER — UNIVERSAL MULTIMODAL CHATBOT MODE")
            print("============================================================")
        router = FreeLLMRouter(env_path=env_path, cache_path=cache_path, force_refresh=force_refresh)
        out_dir = Path(output_dir)

        if preset_sources is not None:
            source_items = list(preset_sources)
        elif is_headless:
            source_items = []
        else:
            source_items = _prompt_chatbot_sources(router)

        source_payload = _load_sources_payload(source_items, router=router)
        if source_payload["file_count"]:
            print(f"[Sources Loaded] {source_payload['summary']} (~{source_payload['tokens']:,} tokens)")
        elif not is_headless:
            print("[Sources Loaded] None (pure chat mode)")

        has_images = bool(source_payload.get("image_paths"))
        if preset_option:
            router.configure_non_interactive(preset_option)
        elif is_headless:
            router.configure_non_interactive("10" if source_payload["file_count"] else "8")
        else:
            router.select_provider_interactive(vision_only=has_images, preset_sources=source_payload if source_payload["file_count"] else None)

    active_format = preset_format if preset_format in SUPPORTED_OUTPUT_FORMATS else "auto"
    history: List[Dict[str, str]] = []
    last_response = ""
    sources_sent_in_history = False

    if not one_shot_prompt and not is_headless:
        print("\n=== Step 3: Chat & Generate Any Output ===")
        print("  Output formats: auto | chat | md | pdf | code | image | videotext | html | docx | json | csv")
        print("  Commands: /sources (add/clear) | /pipeline (switch 1-10) | /out <fmt> | /save <fmt> | /clear | /exit")
        print(f"  Active output mode: {active_format.upper()} | Export folder: {out_dir.resolve()}")

    while True:
        if one_shot_prompt is not None:
            user_prompt = one_shot_prompt.strip()
        else:
            print("")
            user_prompt = router._read_text_or_file_arg("You (prompt, file path, 'paste', or /cmd): ", allow_dir=False)

        if not user_prompt:
            if one_shot_prompt is not None:
                break
            continue

        if user_prompt.lower() in ("/exit", "/quit", "exit", "quit"):
            print("[Chatbot] Session ended.")
            break

        if user_prompt.lower().startswith("/build"):
            from broutegpt_agent import CloudFirstRouter, run_agent_cli

            build_spec = user_prompt[6:].strip()
            if not build_spec:
                build_spec = input("  [Autonomous Build] Describe what you want built: ").strip()
            proj_target = input("  [Autonomous Build] Target project directory [./broutegpt_project]: ").strip() or "./broutegpt_project"
            if build_spec:
                run_agent_cli(
                    build_goal=build_spec,
                    inputs=source_items if source_items else None,
                    project_dir=proj_target,
                    router_override=CloudFirstRouter(router=router),
                )
            continue

        if user_prompt.lower() == "/clear":
            history.clear()
            sources_sent_in_history = False
            print("[Chatbot] Conversation history cleared.")
            continue

        if user_prompt.lower() == "/pipeline":
            router.selected_option = None
            router.select_provider_interactive(
                vision_only=bool(source_payload.get("image_paths")),
                preset_sources=source_payload if source_payload["file_count"] else None,
            )
            continue

        if user_prompt.lower().startswith("/sources"):
            action = input("  [Sources] Enter 'a' to add more, 'c' to clear all, or Enter to inspect: ").strip().lower()
            if action == "c":
                source_items = []
                source_payload = _load_sources_payload(source_items, router=router)
                sources_sent_in_history = False
                print("  [Sources] Cleared.")
            elif action == "a":
                new_items = _prompt_chatbot_sources(router)
                source_items.extend(new_items)
                source_payload = _load_sources_payload(source_items, router=router)
                sources_sent_in_history = False
                print(f"  [Sources] Updated: {source_payload['summary']} (~{source_payload['tokens']:,} tokens)")
            else:
                print(f"  [Sources] Current: {source_payload['summary']} (~{source_payload['tokens']:,} tokens)")
            continue

        if user_prompt.lower().startswith("/out"):
            parts = user_prompt.split(maxsplit=1)
            cand_fmt = parts[1].strip().lower() if len(parts) > 1 else ""
            if cand_fmt in SUPPORTED_OUTPUT_FORMATS:
                active_format = cand_fmt
                print(f"[Chatbot] Active output format set to: {active_format.upper()}")
            else:
                print(f"[Chatbot] Valid formats: {', '.join(SUPPORTED_OUTPUT_FORMATS)}")
            continue

        if user_prompt.lower().startswith("/save"):
            if not last_response:
                print("[Chatbot] No previous response to save yet.")
                continue
            parts = user_prompt.split(maxsplit=1)
            save_fmt = parts[1].strip().lower() if len(parts) > 1 else "md"
            if save_fmt not in SUPPORTED_OUTPUT_FORMATS or save_fmt in ("auto", "chat"):
                print("[Chatbot] Usage: /save <md|pdf|code|image|videotext|html|docx|json|csv>")
                continue
            saved_files = _export_chatbot_output(last_response, save_fmt, out_dir, router)
            for fpath, engine in saved_files:
                print(f"[Saved] {fpath} (via {engine})")
            continue

        inline_detected: List[str] = []
        for tok in re.findall(r"(?:@?['\"]?(?:/|~|\./)[^\s'\"<>|]+['\"]?|https?://[^\s'\"<>|]+)", user_prompt):
            cleaned_tok = tok.strip("'\"").rstrip(".,;:!?")
            if cleaned_tok.startswith(("http://", "https://")):
                inline_detected.append(cleaned_tok)
            else:
                raw_cand = cleaned_tok[1:] if cleaned_tok.startswith("@") else cleaned_tok
                cand_path = Path(os.path.expanduser(raw_cand)) if raw_cand.startswith("~") else Path(raw_cand)
                if cand_path.exists():
                    inline_detected.append(str(cand_path))
        if inline_detected:
            source_items.extend(inline_detected)
            source_payload = _load_sources_payload(source_items, router=router)
            sources_sent_in_history = False
            print(f"[Auto-Attached Source] {source_payload['summary']} (~{source_payload['tokens']:,} tokens)")

        target_fmt = _detect_output_format(user_prompt, active_format)
        sys_instruction = _build_format_system_instruction(target_fmt)

        full_turn_prompt = user_prompt
        attach_imgs: List[str] = []
        if source_payload["file_count"] and not sources_sent_in_history:
            if source_payload["full_text"]:
                full_turn_prompt = (
                    f"Attached Source Material ({source_payload['summary']}):\n\n"
                    f"{source_payload['full_text']}\n\n"
                    f"User Request:\n{user_prompt}"
                )
            attach_imgs = list(source_payload.get("image_paths") or [])
            sources_sent_in_history = True

        with log_ctx:
            response_text, used_model = router.query(
                full_turn_prompt,
                system_prompt=sys_instruction,
                image_paths=attach_imgs if attach_imgs else None,
                history=history,
            )
            last_response = response_text
            history.append({"role": "user", "content": full_turn_prompt})
            history.append({"role": "assistant", "content": response_text})

            saved_files: List[tuple[Path, str]] = []
            if target_fmt != "chat":
                saved_files = _export_chatbot_output(response_text, target_fmt, out_dir, router)

        if json_mode:
            payload = {
                "status": "ok",
                "mode": "chat",
                "model": used_model,
                "output_format": target_fmt,
                "response": response_text,
                "exported_files": [{"path": str(p), "engine": eng} for p, eng in saved_files],
                "usage": router.last_call_usage,
                "duration_sec": round(time.perf_counter() - t_start, 3),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return payload

        if print_only:
            print(response_text)
            for fpath, engine in saved_files:
                print(f"[Exported] {fpath} (via {engine})", file=sys.stderr)
            return {
                "status": "ok",
                "mode": "chat",
                "model": used_model,
                "output_format": target_fmt,
                "response": response_text,
            }

        print(f"\n--- Assistant ({used_model}) [Output: {target_fmt.upper()}] ---")
        print(response_text)
        print("-" * 60)

        for fpath, engine in saved_files:
            print(f"[Exported] {fpath} (via {engine})")

        if one_shot_prompt is not None or bare:
            return {
                "status": "ok",
                "mode": "chat",
                "model": used_model,
                "output_format": target_fmt,
                "response": response_text,
            }
    return None


def run_doctor(
    env_path: str = ".env",
    cache_path: str = DEFAULT_CACHE_FILE,
    output_format: str = "text",
) -> Dict[str, Any]:
    """Inspect Python, Git, OS sandbox, API keys, local Ollama, and cached model readiness."""
    _load_env_file(env_path)
    sys_os = platform.system()

    git_bin = shutil.which("git")
    git_ver = ""
    if git_bin:
        try:
            git_ver = subprocess.run([git_bin, "--version"], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            git_ver = "available"

    sandbox_exec_bin = shutil.which("sandbox-exec")
    bwrap_bin = shutil.which("bwrap")
    no_sandbox_env = os.environ.get("BROUTEGPT_NO_SANDBOX", "").strip() in ("1", "true", "yes")
    if sys_os == "Darwin":
        sandbox_backend = "macos-seatbelt" if sandbox_exec_bin else "none"
    elif sys_os == "Linux":
        sandbox_backend = "linux-bwrap" if bwrap_bin else "none"
    else:
        sandbox_backend = "none"

    provider_keys = {}
    for env_var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "HF_API_KEY"):
        val = os.environ.get(env_var, "")
        configured = _is_valid_key(val)
        provider_keys[env_var] = {
            "configured": configured,
            "preview": f"{val[:4]}...{val[-2:]}" if (configured and len(val) >= 8) else ("set" if configured else "missing"),
        }

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_reachable = False
    ollama_model_count = 0
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=1.5)
        if r.ok:
            ollama_reachable = True
            ollama_model_count = len(r.json().get("models", []))
    except Exception:
        pass

    cache_file = Path(cache_path)
    cache_info: Dict[str, Any] = {
        "path": str(cache_file),
        "exists": cache_file.exists(),
        "total_models": 0,
        "by_option": {},
        "has_role_config": False,
    }
    if cache_file.exists():
        try:
            raw_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            m_by_opt = raw_cache.get("models_by_option") if isinstance(raw_cache, dict) and "models_by_option" in raw_cache else raw_cache
            if isinstance(m_by_opt, dict):
                counts = {k: len(v) for k, v in m_by_opt.items() if isinstance(v, list)}
                cache_info["by_option"] = counts
                cache_info["total_models"] = sum(counts.values())
            if isinstance(raw_cache, dict) and raw_cache.get("role_config"):
                cache_info["has_role_config"] = True
        except Exception as exc:
            cache_info["error"] = str(exc)

    optional_pkgs: Dict[str, bool] = {}
    for mod_name in ("requests", "dotenv", "fitz", "weasyprint", "docx", "openpyxl"):
        try:
            __import__(mod_name)
            optional_pkgs[mod_name] = True
        except ImportError:
            optional_pkgs[mod_name] = False

    healthy = bool(git_bin and optional_pkgs.get("requests", False))
    report: Dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "healthy": healthy,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "os": sys_os,
        "git": {
            "available": bool(git_bin),
            "version": git_ver or "missing",
        },
        "sandbox": {
            "backend": sandbox_backend,
            "disabled_by_env": no_sandbox_env,
        },
        "providers": provider_keys,
        "ollama": {
            "base_url": ollama_url,
            "reachable": ollama_reachable,
            "models": ollama_model_count,
        },
        "model_cache": cache_info,
        "optional_packages": optional_pkgs,
    }

    if str(output_format).strip().lower() == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=== broutegpt --doctor Environment Diagnostics ===")
        print(f"  Status          : {report['status'].upper()}")
        print(f"  Python          : {report['python']['version']} ({report['python']['executable']})")
        print(f"  OS / Sandbox    : {sys_os} | backend={sandbox_backend} (disabled_by_env={no_sandbox_env})")
        print(f"  Git             : {report['git']['version']}")
        print(f"  Ollama Local    : {'REACHABLE' if ollama_reachable else 'unreachable'} ({ollama_url}, {ollama_model_count} model(s))")
        print(f"  Model Cache     : {cache_info['total_models']} cached model(s) at {cache_file} (role_config={cache_info['has_role_config']})")
        print("  API Keys        :")
        for k_name, k_meta in provider_keys.items():
            print(f"    - {k_name:<20}: {'CONFIGURED (' + k_meta['preview'] + ')' if k_meta['configured'] else 'missing'}")
        pkg_str = ", ".join(f"{k}={'ok' if v else 'missing'}" for k, v in optional_pkgs.items())
        print(f"  Packages        : {pkg_str}")

    return report


def _read_piped_stdin() -> str:
    try:
        if sys.stdin is not None and not sys.stdin.isatty():
            return sys.stdin.read().strip()
    except Exception:
        pass
    return ""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="broutegpt — Multi-Provider Free LLM Router, Multimodal Chatbot & Autonomous DP Coding Agent"
    )
    parser.add_argument("--sources", nargs="*", default=None, help="Source files, folders, or URLs to preload (chatbot mode)")
    parser.add_argument("--option", default=None, help="Preselect pipeline option 1-10 or 'provider:model_id'")
    parser.add_argument("--format", dest="out_format", default="auto", choices=SUPPORTED_OUTPUT_FORMATS, help="Target export artifact format")
    parser.add_argument("--out-dir", default="chatbot_outputs", help="Directory where exported outputs are saved")
    parser.add_argument("-p", "--prompt", default=None, help="One-shot prompt (skips interactive chat loop)")
    parser.add_argument("--print", dest="print_mode", action="store_true", help="Headless print mode: execute non-interactively, print response cleanly, and exit")
    parser.add_argument("--output-format", choices=("text", "json"), default="text", help="Stdout output format: text (default) or json")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Shorthand for --output-format json")
    parser.add_argument("--bare", action="store_true", help="CI fast-start mode: skip interactive menus and .broutegpt/MEMORY.md / AGENTS.md rule auto-loading")
    parser.add_argument("--plan", action="store_true", help="Read-only planning mode: generate and print dp_plan.json without executing DP steps")
    parser.add_argument("--doctor", action="store_true", help="Run environment, API key, Git, sandbox, and model cache diagnostics")
    parser.add_argument("--mcp", action="store_true", help="Run broutegpt as a stdio JSON-RPC 2.0 MCP server")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached free models")

    # Autonomous Zero-Question DP Builder & Forensics Flags
    parser.add_argument("--build", nargs="?", const="", default=None, help="Run autonomous zero-question DP project builder for this specification")
    parser.add_argument("--inputs", nargs="*", default=None, help="Input files/folders to stage into sample_inputs/ for --build")
    parser.add_argument("--project-dir", default="./broutegpt_project", help="Target workspace directory for --build / forensics")
    parser.add_argument("--max-dps", type=int, default=6, help="Maximum number of Decision Points to plan (default: 6)")
    parser.add_argument("--max-steps", type=int, default=14, help="Maximum build/debug steps per DP (default: 14)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted autonomous DP build in --project-dir")
    parser.add_argument("--timeline", action="store_true", help="Print shadow-Git checkpoint & model timeline for --project-dir")
    parser.add_argument("--inspect-step", type=int, default=None, help="Inspect unified diff & model forensics for a checkpoint step ID")
    parser.add_argument("--rollback", type=int, default=None, help="Rollback workspace files and/or context to a checkpoint step ID")
    parser.add_argument("--scope", choices=("both", "code", "chat"), default=None, help="Rollback scope: both (default), code (files only), or chat (events only)")
    parser.add_argument("--branch", type=int, default=None, help="Fork a new session branch directory from a checkpoint step ID")
    parser.add_argument("--branch-name", default=None, help="Optional branch name for --branch")
    parser.add_argument("--no-context-rollback", action="store_true", help="With --rollback, restore files only without rewinding event context")
    parser.add_argument("--strict-sandbox", action="store_true", help="Enable stricter sandbox restrictions (blocks network access inside sandbox)")
    parser.add_argument("--auto-guard", action="store_true", help="Run fast free-model safety classifier on unapproved shell commands")
    args = parser.parse_args(argv)

    cli_out_format = "json" if args.json_output else args.output_format

    if args.mcp:
        from broutegpt_agent.mcp_bridge import run_mcp_server

        run_mcp_server(project_dir=args.project_dir)
        sys.exit(0)

    if args.doctor:
        diag = run_doctor(output_format=cli_out_format)
        sys.exit(0 if diag.get("healthy") else 1)

    piped_stdin = _read_piped_stdin()

    if args.plan:
        from broutegpt_agent.dp_orchestrator import DPOrchestrator
        from broutegpt_agent.router_adapter import CloudFirstRouter

        raw_goal = args.build if (args.build is not None and args.build != "") else (args.prompt or "")
        if piped_stdin and raw_goal:
            plan_goal = f"{raw_goal}\n\nPiped Context:\n{piped_stdin}"
        else:
            plan_goal = raw_goal or piped_stdin
        if not plan_goal.strip():
            sys.exit("Error: --plan requires a goal via --build, -p/--prompt, or piped stdin.")

        log_ctx = contextlib.redirect_stdout(sys.stderr) if cli_out_format == "json" else contextlib.nullcontext()
        with log_ctx:
            base_router = FreeLLMRouter(force_refresh=args.refresh)
            if args.option:
                base_router.configure_non_interactive(args.option)
            elif args.bare or args.print_mode or cli_out_format == "json" or not sys.stdin.isatty():
                base_router.configure_non_interactive("deep")
            c_router = CloudFirstRouter(router=base_router)
            orch = DPOrchestrator(
                project_dir=args.project_dir,
                router=c_router,
                max_dps=args.max_dps,
                max_steps_per_dp=args.max_steps,
                strict_sandbox=args.strict_sandbox,
                auto_guard=args.auto_guard,
                bare=args.bare,
            )
            plan_obj = orch.plan_only(goal=plan_goal, inputs=args.inputs)
        if cli_out_format == "json":
            print(json.dumps(plan_obj, indent=2, ensure_ascii=False))
        sys.exit(0)

    if any([
        args.build is not None,
        args.resume,
        args.timeline,
        args.inspect_step is not None,
        args.rollback is not None,
        args.branch is not None,
    ]):
        resolved_build_goal = args.build
        if resolved_build_goal is not None:
            if piped_stdin and resolved_build_goal:
                resolved_build_goal = f"{resolved_build_goal}\n\nPiped Context:\n{piped_stdin}"
            elif piped_stdin and not resolved_build_goal:
                resolved_build_goal = piped_stdin

        if args.bare and resolved_build_goal is not None:
            from broutegpt_agent.dp_orchestrator import DPOrchestrator
            from broutegpt_agent.router_adapter import CloudFirstRouter

            log_ctx = contextlib.redirect_stdout(sys.stderr) if cli_out_format == "json" else contextlib.nullcontext()
            with log_ctx:
                base_router = FreeLLMRouter(force_refresh=args.refresh)
                base_router.configure_non_interactive(args.option or "10")
                c_router = CloudFirstRouter(router=base_router)
                orch = DPOrchestrator(
                    project_dir=args.project_dir,
                    router=c_router,
                    max_dps=args.max_dps,
                    max_steps_per_dp=args.max_steps,
                    strict_sandbox=args.strict_sandbox,
                    auto_guard=args.auto_guard,
                    bare=True,
                )
                build_rep = orch.run(goal=resolved_build_goal, inputs=args.inputs, resume=args.resume)
            if cli_out_format == "json":
                print(json.dumps(build_rep, indent=2, ensure_ascii=False))
        else:
            from broutegpt_agent import run_agent_cli

            resolved_scope = args.scope or ("code" if args.no_context_rollback else "both")
            log_ctx = contextlib.redirect_stdout(sys.stderr) if cli_out_format == "json" else contextlib.nullcontext()
            with log_ctx:
                run_agent_cli(
                    build_goal=resolved_build_goal,
                    inputs=args.inputs,
                    project_dir=args.project_dir,
                    max_dps=args.max_dps,
                    max_steps=args.max_steps,
                    resume=args.resume,
                    show_timeline=args.timeline,
                    inspect_step_id=args.inspect_step,
                    rollback_step_id=args.rollback,
                    restore_context=(resolved_scope in ("both", "chat")),
                    rollback_scope=resolved_scope,
                    branch_step_id=args.branch,
                    branch_name=args.branch_name,
                    preset_option=args.option,
                    force_refresh=args.refresh,
                    strict_sandbox=args.strict_sandbox,
                    auto_guard=args.auto_guard,
                )
            if cli_out_format == "json":
                rep_file = Path(args.project_dir).expanduser().resolve() / ".broutegpt" / "build_report.json"
                if rep_file.is_file():
                    print(rep_file.read_text(encoding="utf-8"))
                else:
                    print(json.dumps({"status": "ok", "project_dir": str(Path(args.project_dir).expanduser().resolve())}, indent=2))

    else:
        effective_prompt = args.prompt
        if piped_stdin and effective_prompt:
            effective_prompt = f"Piped Input:\n{piped_stdin}\n\nPrompt:\n{effective_prompt}"
        elif piped_stdin and not effective_prompt:
            effective_prompt = piped_stdin

        run_chatbot_cli(
            force_refresh=args.refresh,
            preset_sources=args.sources,
            preset_option=args.option,
            preset_format=args.out_format,
            output_dir=args.out_dir,
            one_shot_prompt=effective_prompt,
            print_only=args.print_mode,
            bare=args.bare,
            cli_output_format=cli_out_format,
        )

