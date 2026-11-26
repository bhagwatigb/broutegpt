import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from broutegpt import (
    FreeLLMRouter,
    _estimate_model_tps,
    _is_deep_reasoning_model,
    _sort_models,
    _sort_models_by_speed,
)

NON_CODING_MODEL_KEYWORDS = (
    "image",
    "imagen",
    "lyria",
    "nano-banana",
    "transcribe",
    "whisper",
    "tts",
    "audio",
    "nvclip",
    "video-detector",
    "content-safety",
    "gliner",
    "guard",
    "moderation",
    "embed",
    "rerank",
)


def _clear_status_line(is_tty: bool) -> None:
    if is_tty:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def _write_status_line(msg: str, is_tty: bool) -> None:
    if is_tty:
        cols = shutil.get_terminal_size((100, 24)).columns
        clipped = msg[: max(24, cols - 1)]
        sys.stdout.write(f"\r\033[2K{clipped}")
        sys.stdout.flush()


class CloudFirstRouter:
    """
    Wraps FreeLLMRouter so all eligible cloud models (Gemini, Groq, OpenRouter,
    NVIDIA NIM, Hugging Face) are tried first across tiered strategies, and
    local Ollama is used STRICTLY as a last-resort fallback when every eligible
    cloud model has failed or entered cooldown.

    Transient routing/fallback attempts overwrite a single terminal line and
    auto-erase once a model succeeds so console output stays clean.
    """

    def __init__(self, router: Optional[FreeLLMRouter] = None, **router_kwargs: Any) -> None:
        self.router = router if router is not None else FreeLLMRouter(**router_kwargs)
        self.call_log: List[Dict[str, Any]] = []
        self._provider_cooldown_until: Dict[str, float] = {}

    def _is_cloud(self, model: Dict[str, Any]) -> bool:
        prov = str(model.get("provider", "")).strip().lower()
        if prov == "ollama":
            return False
        if model.get("image_gen"):
            return False
        mid = str(model.get("id", "")).lower()
        if any(bad in mid for bad in NON_CODING_MODEL_KEYWORDS):
            return False
        return True

    def _all_cloud_chat_models(self, vision_only: bool = False) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for opt_key, items in self.router.models_by_option.items():
            if opt_key == "1":
                continue
            for m in items:
                if not m.get("chat", True) or not self._is_cloud(m):
                    continue
                if vision_only and not m.get("vision"):
                    continue
                entry = dict(m)
                entry["option"] = opt_key
                entry["eff_tps"] = float(
                    entry.get("measured_tps")
                    or entry.get("est_tps")
                    or _estimate_model_tps(entry.get("provider", ""), entry.get("id", ""))
                )
                entry["total_capacity"] = int(entry.get("input_limit", 0)) + int(entry.get("output_limit", 0))
                out.append(entry)
        return out

    def get_cloud_candidates(
        self,
        prompt: str,
        tier: str = "smart",
        system_prompt: Optional[str] = None,
        vision_only: bool = False,
        explicit_max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        task_plan: Optional[Dict[str, Any]] = None
        seen: set[tuple[str, str]] = set()
        ordered: List[Dict[str, Any]] = []

        def _append_unique(items: List[Dict[str, Any]]) -> None:
            for m in items:
                if not self._is_cloud(m) or not m.get("chat", True):
                    continue
                if vision_only and not m.get("vision"):
                    continue
                key = (m["provider"], m["id"])
                if key not in seen:
                    seen.add(key)
                    ordered.append(dict(m))

        all_cloud = self._all_cloud_chat_models(vision_only=vision_only)
        lookup = {(m["provider"], m["id"]): m for m in all_cloud}

        if tier == "smart":
            is_tty = sys.stdout.isatty()
            _write_status_line("[Router] Estimating token budget & ranking cloud models...", is_tty)
            history_text = "\n".join(m.get("content", "") for m in (history or []))
            sizing_prompt = f"{history_text}\n{prompt}".strip() if history_text else prompt

            saved_ollama = self.router.models_by_option.get("1", [])
            orig_call = self.router._call_model

            def _cooldown_aware_probe(prov: str, mid: str, *a: Any, **kw: Any) -> str:
                if self.router._cooldown_until.get((prov, mid), 0.0) > time.time():
                    raise RuntimeError(f"429 Cooldown active for {prov}:{mid}")
                try:
                    return orig_call(prov, mid, *a, **kw)
                except Exception as exc:
                    err_s = str(exc)
                    if "429" in err_s or "rate" in err_s.lower():
                        self.router._cooldown_until[(prov, mid)] = time.time() + 60.0
                    elif any(c in err_s for c in ("400", "401", "403", "404")):
                        self.router._cooldown_until[(prov, mid)] = time.time() + 600.0
                    raise

            self.router.models_by_option["1"] = []
            self.router._call_model = _cooldown_aware_probe
            try:
                task_plan = self.router.recommend_models_for_task(
                    sizing_prompt,
                    system_prompt=system_prompt,
                    explicit_max_tokens=explicit_max_tokens,
                    quiet=True,
                )
            finally:
                self.router._call_model = orig_call
                self.router.models_by_option["1"] = saved_ollama
                _clear_status_line(is_tty)

            _append_unique(task_plan.get("ranked_candidates") or [])
            _append_unique(_sort_models_by_speed(all_cloud))

        elif tier == "deep":
            for ref in self.router.role_config.get("deep_synthesis") or []:
                key = (ref.get("provider", ""), ref.get("id", ""))
                if key in lookup:
                    _append_unique([lookup[key]])
            deep_pool = [m for m in _sort_models(all_cloud) if _is_deep_reasoning_model(m)]
            _append_unique(deep_pool)
            _append_unique(_sort_models(all_cloud))

        elif tier == "speed":
            _append_unique(_sort_models_by_speed(all_cloud))

        else:
            _append_unique(_sort_models(all_cloud))

        return ordered, task_plan

    def get_ollama_candidates(self, vision_only: bool = False) -> List[Dict[str, Any]]:
        """Returns local Ollama models strictly for last-resort fallback."""
        pool = [
            dict(m)
            for m in self.router.models_by_option.get("1", [])
            if m.get("chat", True) and (not vision_only or m.get("vision"))
        ]

        def _coder_priority(m: Dict[str, Any]) -> tuple[int, float]:
            mid = m.get("id", "").lower()
            is_coder = 0 if any(k in mid for k in ("coder", "qwen3", "qwen2.5", "llama3.1", "deepseek")) else 1
            tps = float(m.get("measured_tps") or m.get("est_tps") or _estimate_model_tps("ollama", mid))
            return (is_coder, -tps)

        return sorted(pool, key=_coder_priority)

    def _ensure_ollama_running(self) -> None:
        """Starts `ollama serve` in the background on macOS if last-resort fallback is triggered."""
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            return
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if resp.ok:
                return
        except Exception:
            pass
        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            for _ in range(6):
                time.sleep(0.5)
                try:
                    if requests.get("http://localhost:11434/api/tags", timeout=1.0).ok:
                        return
                except Exception:
                    continue
        except Exception:
            pass

    def _try_candidate_list(
        self,
        candidates: List[Dict[str, Any]],
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        task_plan: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
        validator: Optional[Callable[[str], Optional[str]]],
        image_paths: Optional[List[str]],
        stage_label: str,
    ) -> tuple[Optional[str], Optional[str], List[str]]:
        now = time.time()
        active = [
            m for m in candidates
            if self.router._cooldown_until.get((m["provider"], m["id"]), 0.0) <= now
            and self._provider_cooldown_until.get(m["provider"], 0.0) <= now
        ]
        if not active:
            return None, None, [f"[{stage_label}] All {len(candidates)} candidate(s) currently in cooldown."]

        is_tty = sys.stdout.isatty()
        errors: List[str] = []

        for idx, model in enumerate(active, 1):
            prov = model["provider"]
            mid = model["id"]
            if self._provider_cooldown_until.get(prov, 0.0) > time.time():
                continue

            eff_max = max_tokens
            if task_plan is not None:
                target_out = max_tokens or max(int(task_plan["safe_output_tokens"]), 4096)
                out_lim = int(model.get("output_limit") or target_out)
                eff_max = min(target_out, out_lim)

            t0 = time.perf_counter()
            _write_status_line(
                f"[Routing:{stage_label}] Trying {idx}/{len(active)}: {prov}:{mid} ...",
                is_tty,
            )
            try:
                text = self.router._call_model(
                    prov,
                    mid,
                    prompt,
                    system_prompt,
                    None,
                    temperature,
                    eff_max,
                    image_paths=image_paths,
                    history=history,
                )
                if any(
                    text.count(text[i : i + 25]) > 8
                    for i in range(0, min(len(text), 4000), 25)
                    if len(text[i : i + 25].strip()) >= 15
                ):
                    raise ValueError("Degenerate substring repetition loop detected")

                if validator is not None:
                    val_err = validator(text)
                    if val_err:
                        raise ValueError(f"Validator rejected output: {val_err}")

                elapsed = round(time.perf_counter() - t0, 2)
                self.call_log.append({
                    "timestamp": time.time(),
                    "stage": stage_label,
                    "model": f"{prov}:{mid}",
                    "status": "ok",
                    "latency_sec": elapsed,
                })

                # Erase all transient routing/fallback status text and print only the winning model
                _clear_status_line(is_tty)
                reroute_badge = f" (rerouted after {len(errors)} fallback{'s' if len(errors) != 1 else ''})" if errors else ""
                print(f"[Model:{stage_label}] {prov}:{mid} ({elapsed}s){reroute_badge}")
                return text, f"{prov}:{mid}", errors

            except Exception as exc:
                err_str = str(exc)
                elapsed = round(time.perf_counter() - t0, 2)
                if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                    self.router._cooldown_until[(prov, mid)] = time.time() + 60.0
                elif "Validator rejected output" in err_str:
                    self.router._cooldown_until[(prov, mid)] = time.time() + 120.0
                elif any(code in err_str for code in ("400", "404")):
                    self.router._cooldown_until[(prov, mid)] = time.time() + 600.0
                elif any(code in err_str for code in ("401", "403", "No connection adapters")):
                    self._provider_cooldown_until[prov] = time.time() + 300.0

                self.call_log.append({
                    "timestamp": time.time(),
                    "stage": stage_label,
                    "model": f"{prov}:{mid}",
                    "status": "error",
                    "error": err_str[:240],
                    "latency_sec": elapsed,
                })
                errors.append(f"{prov}:{mid} -> {err_str}")
                short_err = err_str.splitlines()[0][:65]
                _write_status_line(
                    f"[Fallback:{stage_label}] {prov}:{mid} ({short_err}) -> trying {idx + 1}/{len(active)}...",
                    is_tty,
                )

        _clear_status_line(is_tty)
        return None, None, errors

    def query_autonomous(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "smart",
        temperature: float = 0.15,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        validator: Optional[Callable[[str], Optional[str]]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> tuple[str, str]:
        """
        Zero-question autonomous query:
          1. Exhausts all active cloud models for the selected tier.
          2. Only if EVERY cloud candidate fails or is in cooldown, falls back to local Ollama.
        """
        vision_only = bool(image_paths)
        cloud_list, task_plan = self.get_cloud_candidates(
            prompt=prompt,
            tier=tier,
            system_prompt=system_prompt,
            vision_only=vision_only,
            explicit_max_tokens=max_tokens,
            history=history,
        )

        text, used_model, cloud_errors = self._try_candidate_list(
            candidates=cloud_list,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            task_plan=task_plan,
            history=history,
            validator=validator,
            image_paths=image_paths,
            stage_label=f"CLOUD:{tier}",
        )
        if text is not None and used_model is not None:
            return text, used_model

        ollama_list = self.get_ollama_candidates(vision_only=vision_only)
        if not ollama_list:
            raise RuntimeError(
                "All eligible cloud models failed/429 and no local Ollama models are available.\n"
                + "\n".join(cloud_errors[-10:])
            )

        print(
            f"[Fallback -> Local Ollama] Cloud pool exhausted ({len(cloud_list)} models) -> "
            f"trying local Ollama ({len(ollama_list)} model(s))..."
        )
        self._ensure_ollama_running()
        text, used_model, ollama_errors = self._try_candidate_list(
            candidates=ollama_list,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            task_plan=None,
            history=history,
            validator=validator,
            image_paths=image_paths,
            stage_label="OLLAMA:LAST_RESORT",
        )
        if text is not None and used_model is not None:
            return text, used_model

        raise RuntimeError(
            "Both cloud pool and last-resort local Ollama models failed:\n"
            + "\n".join((cloud_errors[-5:] + ollama_errors[-5:]))
        )
