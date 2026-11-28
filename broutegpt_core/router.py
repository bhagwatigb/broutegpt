import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import (
    DEFAULT_CACHE_FILE,
    PROVIDER_RATE_INFO,
    _is_valid_key,
    _redact_secrets,
    load_all_api_keys,
)
from .sources import _estimate_text_tokens, _load_sources_payload


def _estimate_model_tps(provider: str, model_id: str) -> float:
    mid = model_id.lower()
    if provider == "groq":
        if any(k in mid for k in ("8b", "instant", "gemma", "small")):
            return 750.0
        if "70b" in mid or "versatile" in mid:
            return 320.0
        return 450.0
    if provider == "gemini":
        if "lite" in mid or "8b" in mid:
            return 240.0
        if "flash" in mid:
            return 175.0
        if "gemma" in mid:
            return 140.0
        return 85.0
    if provider == "nvidia":
        if any(k in mid for k in ("8b", "7b", "mini", "small", "nano")):
            return 145.0
        return 85.0
    if provider == "openrouter":
        if any(k in mid for k in ("flash", "8b", "7b", "mini", "small")):
            return 95.0
        return 55.0
    if provider == "huggingface":
        if any(k in mid for k in ("8b", "7b", "mini", "small")):
            return 75.0
        return 45.0
    if provider == "ollama":
        if any(k in mid for k in ("1b", "3b", "mini", "tiny")):
            return 65.0
        if any(k in mid for k in ("7b", "8b")):
            return 35.0
        return 18.0
    return 50.0


def _sort_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        models,
        key=lambda m: (
            m.get("input_limit", 0) + m.get("output_limit", 0),
            m.get("output_limit", 0),
            m.get("input_limit", 0),
        ),
        reverse=True,
    )


def _sort_models_by_speed(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        models,
        key=lambda m: (
            float(m.get("measured_tps") or m.get("est_tps") or _estimate_model_tps(m.get("provider", ""), m.get("id", ""))),
            m.get("output_limit", 0),
            m.get("input_limit", 0),
        ),
        reverse=True,
    )


def _heuristic_output_estimate(prompt: str, input_tokens: int) -> Dict[str, Any]:
    low = prompt.lower()
    if any(k in low for k in ("yes or no", "one word", "single word", "true or false", "classify", "label only")):
        return {"estimated_output_tokens": 160, "reasoning_depth": "light", "rationale": "Short classification/single-value prompt"}
    if any(k in low for k in ("brief", "short", "concise", "tldr", "tl;dr", "one sentence", "two sentences", "few bullet", "summarize")):
        est = max(450, min(int(input_tokens * 0.25), 2500))
        return {"estimated_output_tokens": est, "reasoning_depth": "light", "rationale": "Concise summary/short-form synthesis requested"}
    if any(k in low for k in ("exhaustive", "comprehensive", "detailed", "in-depth", "step-by-step", "full code", "complete script", "entire", "essay", "report", "all sections", "rewrite", "zero-loss", "union-merge")):
        est = max(3000, min(int(input_tokens * 0.90), 65536))
        return {"estimated_output_tokens": est, "reasoning_depth": "deep", "rationale": "Zero-loss merge or exhaustive synthesis across full input payload"}
    est = max(900, min(int(input_tokens * 0.45), 6000))
    return {"estimated_output_tokens": est, "reasoning_depth": "moderate", "rationale": "Standard multi-paragraph or structured response"}


def _model_fits_budget(model: Dict[str, Any], safe_in: int, safe_out: int) -> bool:
    prov = model.get("provider", "")
    in_lim = int(model.get("input_limit") or 0)
    out_lim = int(model.get("output_limit") or 0)
    if in_lim <= 0 or out_lim <= 0:
        return False
    if out_lim < safe_out:
        return False
    if prov == "gemini":
        return in_lim >= safe_in
    return in_lim >= (safe_in + safe_out)


def _deep_reasoning_score(model: Dict[str, Any]) -> int:
    mid = model.get("id", "").lower()
    tokens = set(re.split(r"[-_/:]+", mid))
    if tokens & {"0.5b", "1b", "1.5b", "2b", "3b", "4b", "7b", "8b", "9b", "12b", "lite", "mini", "nano", "tiny", "instant"}:
        return 0
    if (tokens & {"r1", "qwq", "o1", "o3"}) or any(k in mid for k in ("thinking", "reasoner", "235b", "405b", "gpt-oss-120b")):
        return 4
    if any(k in mid for k in ("120b", "70b", "72b", "llama-3.3", "llama-4", "maverick", "kimi-k2", "glm-4", "qwen3", "32b", "27b", "large", "plus", "versatile")):
        return 3
    if any(k in mid for k in ("flash", "scout", "24b", "17b", "gemma-3")):
        return 2
    return 1


def _is_deep_reasoning_model(model: Dict[str, Any]) -> bool:
    return _deep_reasoning_score(model) >= 2


def _encode_image(image_path: str) -> tuple[str, str]:
    raw_bytes = Path(image_path).read_bytes()
    mime_type, _ = mimetypes.guess_type(image_path)
    return base64.b64encode(raw_bytes).decode("utf-8"), mime_type or "image/png"


class FreeLLMRouter:
    OPTIONS = {
        "1": "Local Ollama Models",
        "2": "Google Gemini (Free Tier)",
        "3": "Groq (Free Tier)",
        "4": "OpenRouter (Free Models)",
        "5": "Other Free APIs (NVIDIA NIM / Hugging Face)",
    }

    def __init__(
        self,
        env_path: str = ".env",
        cache_path: str = DEFAULT_CACHE_FILE,
        force_refresh: bool = False,
        timeout: int = 90,
    ) -> None:
        keys = load_all_api_keys(env_path)
        for k, v in keys.items():
            os.environ[k] = v
        self.cache_path = Path(cache_path)
        self.timeout = timeout
        self.selected_option: Optional[str] = None
        self.custom_model: Optional[Dict[str, str]] = None
        self.models_by_option: Dict[str, List[Dict[str, Any]]] = {}
        self.role_config: Dict[str, List[Dict[str, str]]] = {}
        self.eligible_models: List[Dict[str, Any]] = []
        self.last_task_plan: Optional[Dict[str, Any]] = None
        self.last_call_usage: Dict[str, Any] = {}
        self._cooldown_until: Dict[tuple[str, str], float] = {}

        loaded_ok = False
        if not force_refresh and self.cache_path.exists():
            try:
                raw_cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(raw_cache, dict) and "models_by_option" in raw_cache:
                    self.models_by_option = raw_cache.get("models_by_option") or {}
                    self.role_config = raw_cache.get("role_config") or {}
                elif isinstance(raw_cache, dict):
                    self.models_by_option = raw_cache
                    self.role_config = {}
                loaded_ok = bool(self.models_by_option)
            except Exception:
                self.models_by_option = {}
                self.role_config = {}

        if not loaded_ok or force_refresh:
            self.discover_and_cache_models()
        elif not self.role_config:
            self.run_reconfiguration_drive()

    def _save_cache(self) -> None:
        try:
            payload = {
                "models_by_option": self.models_by_option,
                "role_config": self.role_config,
            }
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[Router] Cache save warning: {_redact_secrets(exc)}")

    def _all_discovered_models(self) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        for opt_key, items in self.models_by_option.items():
            for m in items:
                entry = dict(m)
                entry["option"] = opt_key
                all_items.append(entry)
        return all_items

    def _build_default_role_config(self, catalog: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
        chat_models = [m for m in catalog if m.get("chat", True)]
        speed_sorted = _sort_models_by_speed(chat_models)
        cap_sorted = _sort_models(chat_models)

        vision_models = _sort_models_by_speed([m for m in chat_models if m.get("vision")])
        image_gen_models = [
            m for m in catalog
            if m.get("image_gen") or any(k in m.get("id", "").lower() for k in ("flux", "sdxl", "stable-diffusion", "image", "recraft", "kolors", "playground"))
        ]
        audio_models = [
            m for m in catalog
            if m.get("modality") == "audio_transcription" or any(k in m.get("id", "").lower() for k in ("whisper", "asr", "stt"))
        ]
        deep_ranked = sorted(
            [m for m in catalog if m.get("chat", True) and _is_deep_reasoning_model(m)],
            key=lambda m: (_deep_reasoning_score(m), m.get("output_limit", 0), m.get("input_limit", 0)),
            reverse=True,
        )
        deep_models = deep_ranked or cap_sorted

        def _refs(items: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            seen = set()
            for item in items:
                key = (item["provider"], item["id"])
                if key not in seen:
                    seen.add(key)
                    out.append({"provider": item["provider"], "id": item["id"]})
                if len(out) >= limit:
                    break
            return out

        return {
            "image_transcription": _refs(vision_models, 6),
            "image_generation": _refs(image_gen_models, 5),
            "audio_transcription": _refs(audio_models, 4),
            "fast_triage": _refs(speed_sorted, 5),
            "deep_synthesis": _refs(deep_models, 5),
        }

    def run_reconfiguration_drive(self) -> Dict[str, List[Dict[str, str]]]:
        catalog = self._all_discovered_models()
        defaults = self._build_default_role_config(catalog)
        if not catalog:
            self.role_config = defaults
            self._save_cache()
            return self.role_config

        print("\n[Reconfiguration Drive] Analyzing freshly discovered free models to auto-configure specialized roles...")
        valid_lookup = {(m["provider"], m["id"]): m for m in catalog}

        catalog_lines: List[str] = []
        for m in catalog[:90]:
            tags: List[str] = []
            if m.get("chat", True):
                tags.append("chat")
            if m.get("vision"):
                tags.append("vision")
            if m.get("image_gen"):
                tags.append("image_gen")
            if m.get("modality") == "audio_transcription":
                tags.append("audio_transcription")
            tps = float(m.get("measured_tps") or m.get("est_tps") or _estimate_model_tps(m["provider"], m["id"]))
            catalog_lines.append(
                f"- {m['provider']}:{m['id']} | tags={','.join(tags) or 'general'} | in={m.get('input_limit', 0)} | out={m.get('output_limit', 0)} | ~{tps:.0f}tok/s"
            )

        reconfig_prompt = (
            "You are configuring an AI model router using ONLY the currently discovered free models listed below.\n"
            "Select the best-suited ranked models from this exact list for each functional need:\n"
            "1. image_transcription: Vision-capable models best suited for OCR, diagram reading, and transcribing images.\n"
            "2. image_generation: Models best suited for image generation (must have image_gen tag if present; otherwise empty list []).\n"
            "3. audio_transcription: Speech-to-text models best suited for transcribing audio/video (must have audio_transcription tag if present; otherwise empty list []).\n"
            "4. fast_triage: Fastest reliable chat models for quick prompt/token sizing.\n"
            "5. deep_synthesis: Strongest high-context reasoning chat models for complex multi-source tasks.\n\n"
            "CURRENT FREE MODELS LIST:\n"
            + "\n".join(catalog_lines)
            + "\n\nReply ONLY with valid JSON matching this exact schema (use exact provider and id strings from the list above):\n"
            '{"image_transcription":[{"provider":"...","id":"..."}],'
            '"image_generation":[{"provider":"...","id":"..."}],'
            '"audio_transcription":[{"provider":"...","id":"..."}],'
            '"fast_triage":[{"provider":"...","id":"..."}],'
            '"deep_synthesis":[{"provider":"...","id":"..."}]}'
        )

        fast_chat = _sort_models_by_speed([m for m in catalog if m.get("chat", True)])
        configured_by = "heuristic-catalog-match"
        parsed_roles: Dict[str, List[Dict[str, str]]] = {}

        for m in fast_chat[:8]:
            prov, mid = m["provider"], m["id"]
            if not _model_fits_budget(m, _estimate_text_tokens(reconfig_prompt) + 64, 512):
                continue
            try:
                raw = self._call_model(prov, mid, reconfig_prompt, None, None, 0.0, 600)
                match = re.search(r"\{[\s\S]*\}", raw)
                if match:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict):
                        for role_name in ("image_transcription", "image_generation", "audio_transcription", "fast_triage", "deep_synthesis"):
                            entries = data.get(role_name) or []
                            cleaned: List[Dict[str, str]] = []
                            seen = set()
                            for item in entries:
                                if not isinstance(item, dict):
                                    continue
                                p_val = str(item.get("provider", "")).strip().lower()
                                id_val = str(item.get("id", "")).strip()
                                if (p_val, id_val) in valid_lookup and (p_val, id_val) not in seen:
                                    meta = valid_lookup[(p_val, id_val)]
                                    if role_name == "image_transcription" and not meta.get("vision"):
                                        continue
                                    if role_name == "audio_transcription" and meta.get("modality") != "audio_transcription":
                                        continue
                                    seen.add((p_val, id_val))
                                    cleaned.append({"provider": p_val, "id": id_val})
                            parsed_roles[role_name] = cleaned
                        configured_by = f"{prov}:{mid}"
                        break
            except Exception:
                continue

        final_roles: Dict[str, List[Dict[str, str]]] = {}
        for role_name, fallback_list in defaults.items():
            chosen = list(parsed_roles.get(role_name) or [])
            seen = {(x["provider"], x["id"]) for x in chosen}
            for fb in fallback_list:
                if (fb["provider"], fb["id"]) not in seen:
                    seen.add((fb["provider"], fb["id"]))
                    chosen.append(fb)
            final_roles[role_name] = chosen

        self.role_config = final_roles
        self._save_cache()

        print(f"[Reconfiguration Drive] Completed via {configured_by}:")
        for role_name, items in self.role_config.items():
            top_str = ", ".join(f"{x['provider']}:{x['id']}" for x in items[:2]) if items else "None discovered (will use dynamic fallback)"
            print(f"  - {role_name:<20}: {top_str}")
        return self.role_config

    def discover_and_cache_models(self) -> None:
        print("[Router] Discovering available free models (filtering out preview/404 endpoints)...")
        discovered = {
            "1": _sort_models(self._discover_ollama()),
            "2": _sort_models(self._discover_gemini()),
            "3": _sort_models(self._discover_groq()),
            "4": _sort_models(self._discover_openrouter()),
            "5": _sort_models(self._discover_other_free()),
        }
        self.models_by_option = discovered
        self._save_cache()
        print(f"[Router] Saved ranked stable model lists to {self.cache_path}")
        self.run_reconfiguration_drive()

    def _discover_ollama(self) -> List[Dict[str, Any]]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        models = []
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            if resp.ok:
                for item in resp.json().get("models", []):
                    name = item.get("name")
                    if name:
                        is_vision = any(k in name.lower() for k in ("llava", "vision", "vl", "minicpm", "gemma3", "qwen2.5vl"))
                        models.append({
                            "id": name,
                            "provider": "ollama",
                            "input_limit": 8192,
                            "output_limit": 4096,
                            "vision": is_vision,
                            "est_tps": _estimate_model_tps("ollama", name),
                        })
        except Exception:
            pass
        return models

    def _discover_gemini(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not _is_valid_key(api_key):
            return []
        models = []
        try:
            resp = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("models", []):
                if "generateContent" not in item.get("supportedGenerationMethods", []):
                    continue
                name = item.get("name", "").replace("models/", "")
                if any(bad in name.lower() for bad in ("embedding", "imagen", "veo", "tts", "aqa", "pro", "preview", "experimental")):
                    continue
                is_vision = "flash" in name.lower() or "gemma-3" in name.lower()
                is_img_gen = "image" in name.lower()
                models.append({
                    "id": name,
                    "provider": "gemini",
                    "input_limit": int(item.get("inputTokenLimit") or 0),
                    "output_limit": int(item.get("outputTokenLimit") or 0),
                    "vision": is_vision,
                    "image_gen": is_img_gen,
                    "chat": True,
                    "modality": "image_generation" if is_img_gen else ("vision" if is_vision else "chat"),
                    "est_tps": _estimate_model_tps("gemini", name),
                })
        except Exception as exc:
            print(f"[Router] Gemini discovery warning: {_redact_secrets(exc)}")
        return models

    def _discover_groq(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("GROQ_API_KEY")
        if not _is_valid_key(api_key):
            return []
        models = []
        try:
            resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                mid = item.get("id", "")
                low = mid.lower()
                if any(bad in low for bad in ("tts", "orpheus", "guard", "prompt-guard", "preview")):
                    continue
                if "whisper" in low:
                    models.append({
                        "id": mid,
                        "provider": "groq",
                        "input_limit": 0,
                        "output_limit": 0,
                        "vision": False,
                        "chat": False,
                        "modality": "audio_transcription",
                        "est_tps": 900.0,
                    })
                    continue

                is_vision = any(k in mid.lower() for k in ("vision", "vl", "scout", "maverick", "qwen3"))
                models.append({
                    "id": mid,
                    "provider": "groq",
                    "input_limit": int(item.get("context_window") or 8192),
                    "output_limit": int(item.get("max_completion_tokens") or 8192),
                    "vision": is_vision,
                    "est_tps": _estimate_model_tps("groq", mid),
                })
        except Exception as exc:
            print(f"[Router] Groq discovery warning: {_redact_secrets(exc)}")
        return models

    def _discover_openrouter(self) -> List[Dict[str, Any]]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not _is_valid_key(api_key):
            return []
        models = []
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                mid = item.get("id", "")
                if any(bad in mid.lower() for bad in ("preview", "beta", "experimental")):
                    continue
                pricing = item.get("pricing", {})
                is_free = mid.endswith(":free") or mid == "openrouter/free" or (
                    str(pricing.get("prompt", "")) == "0" and str(pricing.get("completion", "")) == "0"
                )
                if not is_free:
                    continue
                arch = item.get("architecture") or {}
                in_mods = arch.get("input_modalities") or []
                out_mods = arch.get("output_modalities") or []
                is_vision = ("image" in in_mods) or any(k in mid.lower() for k in ("vl", "vision", "flash", "gemini", "gemma-3", "pixtral", "molmo"))
                is_img_gen = ("image" in out_mods)
                ctx = int(item.get("context_length") or 8192)
                out_lim = int((item.get("top_provider") or {}).get("max_completion_tokens") or min(ctx, 8192))
                models.append({
                    "id": mid,
                    "provider": "openrouter",
                    "input_limit": ctx,
                    "output_limit": out_lim,
                    "vision": is_vision,
                    "image_gen": is_img_gen,
                    "chat": True,
                    "modality": "image_generation" if is_img_gen else ("vision" if is_vision else "chat"),
                    "est_tps": _estimate_model_tps("openrouter", mid),
                })
        except Exception as exc:
            print(f"[Router] OpenRouter discovery warning: {_redact_secrets(exc)}")
        return models

    def _discover_other_free(self) -> List[Dict[str, Any]]:
        models = []
        specs = [
            ("NVIDIA_API_KEY", "nvidia", "https://integrate.api.nvidia.com/v1/models", 32768, 4096),
            ("HF_API_KEY", "huggingface", "https://router.huggingface.co/v1/models", 16384, 4096),
        ]
        for env_var, prov, url, def_in, def_out in specs:
            key = os.getenv(env_var)
            if not _is_valid_key(key):
                continue
            try:
                resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=15)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    mid = item.get("id", "")
                    if not mid or any(bad in mid.lower() for bad in ("embed", "rerank", "guard", "moderation", "parse", "preview")):
                        continue
                    low = mid.lower()
                    is_vision = any(k in low for k in ("vl", "vision", "neva", "kosmos", "phi-3-vision", "phi-3.5-vision"))
                    is_img_gen = any(k in low for k in ("flux", "stable-diffusion", "sdxl", "image"))
                    models.append({
                        "id": mid,
                        "provider": prov,
                        "input_limit": def_in,
                        "output_limit": def_out,
                        "vision": is_vision,
                        "image_gen": is_img_gen,
                        "chat": not is_img_gen,
                        "modality": "image_generation" if is_img_gen else ("vision" if is_vision else "chat"),
                        "est_tps": _estimate_model_tps(prov, mid),
                    })
            except Exception:
                pass
        return models

    def _get_option_models(self, option: str, vision_only: bool = False) -> List[Dict[str, Any]]:
        pool = [m for m in self.models_by_option.get(option, []) if m.get("chat", True)]
        if vision_only:
            pool = [m for m in pool if m.get("vision", False)]
        return pool

    def _get_ranked_models(self, top_n: Optional[int] = 10, vision_only: bool = False) -> List[Dict[str, Any]]:
        all_models: List[Dict[str, Any]] = []
        for opt_key, items in self.models_by_option.items():
            for m in items:
                if not m.get("chat", True):
                    continue
                if vision_only and not m.get("vision"):
                    continue
                entry = dict(m)
                entry["option"] = opt_key
                all_models.append(entry)
        ranked = _sort_models(all_models)
        return ranked[:top_n] if top_n else ranked

    def _get_speed_ranked_models(self, top_n: Optional[int] = 10, vision_only: bool = False) -> List[Dict[str, Any]]:
        all_models: List[Dict[str, Any]] = []
        for opt_key, items in self.models_by_option.items():
            for m in items:
                if not m.get("chat", True):
                    continue
                if vision_only and not m.get("vision"):
                    continue
                entry = dict(m)
                entry["option"] = opt_key
                if "est_tps" not in entry:
                    entry["est_tps"] = _estimate_model_tps(entry.get("provider", ""), entry.get("id", ""))
                all_models.append(entry)
        ranked = _sort_models_by_speed(all_models)
        return ranked[:top_n] if top_n else ranked

    def transcribe_image(
        self,
        image_path: str,
        prompt: str = "Transcribe all visible text, equations, tables, labels, and diagram structure accurately in clean Markdown.",
        temperature: float = 0.1,
    ) -> tuple[str, str]:
        candidates: List[Dict[str, str]] = list(self.role_config.get("image_transcription") or [])
        seen = {(c["provider"], c["id"]) for c in candidates}
        for m in self._get_speed_ranked_models(top_n=None, vision_only=True):
            key = (m["provider"], m["id"])
            if key not in seen:
                seen.add(key)
                candidates.append({"provider": m["provider"], "id": m["id"]})

        if not candidates:
            raise RuntimeError("No vision-capable free models available for image transcription. Configure API keys and press 'r' to refresh.")

        errors: List[str] = []
        for cand in candidates:
            prov, mid = cand["provider"], cand["id"]
            try:
                text = self._call_model(prov, mid, prompt, None, image_path, temperature, None)
                return text, f"{prov}:{mid}"
            except Exception as exc:
                errors.append(f"{prov}:{mid} -> {_redact_secrets(exc)}")
        raise RuntimeError("All configured image_transcription models failed:\n" + "\n".join(errors))

    def compare_top_models(self, top_n: int = 10, vision_only: bool = False) -> None:
        ranked = self._get_ranked_models(top_n=top_n, vision_only=vision_only)
        if not ranked:
            print("[Router] No models cached yet. Add keys in .env and press 'r' to refresh.")
            return
        tag = "Vision " if vision_only else ""
        print(f"\n=== Top {len(ranked)} {tag}Models Side-by-Side (Sorted by Input + Output Token Limits) ===")
        header = f"{'#':<3} {'Opt':<4} {'Provider':<12} {'Model ID':<38} {'Vision':<7} {'In Limit':>10} {'Out Limit':>10} {'Est. Req/Hr':<21}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(ranked, 1):
            prov = m.get("provider", "?")
            rph, _ = PROVIDER_RATE_INFO.get(prov, ("Unknown", ""))
            mid = m.get("id", "")[:37]
            vis = "Yes" if m.get("vision") else "No"
            print(f"{i:<3} {m.get('option','?'):<4} {prov:<12} {mid:<38} {vis:<7} {m.get('input_limit',0):>10,} {m.get('output_limit',0):>10,} {rph:<21}")

    def benchmark_models_speed(self, top_n: int = 10, vision_only: bool = False) -> None:
        candidates = self._get_speed_ranked_models(top_n=top_n, vision_only=vision_only)
        if not candidates:
            print("[Router] No models available to benchmark.")
            return
        print(f"\n[Speed Benchmark] Running live output speed test across Top {len(candidates)} candidate models...")
        test_prompt = "Count from 1 to 40 separated by spaces and briefly state what an API router does in two sentences."
        for idx, m in enumerate(candidates, 1):
            prov, mid, opt_key = m["provider"], m["id"], m["option"]
            print(f"  -> ({idx}/{len(candidates)}) Testing {prov}:{mid} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                out = self._call_model(prov, mid, test_prompt, None, None, 0.1, 120)
                elapsed = max(time.perf_counter() - t0, 0.001)
                est_tokens = max(int(len(out.split()) * 1.35), 1)
                tps = round(est_tokens / elapsed, 1)
                latency_ms = int(elapsed * 1000)
                print(f"{tps} tok/s ({latency_ms} ms)")
                for item in self.models_by_option.get(opt_key, []):
                    if item.get("id") == mid and item.get("provider") == prov:
                        item["measured_tps"] = tps
                        item["latency_ms"] = latency_ms
            except Exception as exc:
                print(f"FAILED ({_redact_secrets(exc)})")
        self._save_cache()

    def compare_models_by_speed(self, top_n: int = 10, vision_only: bool = False) -> None:
        ranked = self._get_speed_ranked_models(top_n=top_n, vision_only=vision_only)
        if not ranked:
            print("[Router] No models cached yet. Add keys in .env and press 'r' to refresh.")
            return
        tag = "Vision " if vision_only else ""
        print(f"\n=== Top {len(ranked)} Fastest {tag}Models Side-by-Side (Sorted by Output Speed) ===")
        header = f"{'#':<3} {'Opt':<4} {'Provider':<12} {'Model ID':<36} {'Speed (Tok/s)':>15} {'Source':<10} {'Out Limit':>10}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(ranked, 1):
            prov = m.get("provider", "?")
            mid = m.get("id", "")[:35]
            if m.get("measured_tps"):
                speed_val = float(m["measured_tps"])
                src = "Measured"
            else:
                speed_val = float(m.get("est_tps") or _estimate_model_tps(prov, m.get("id", "")))
                src = "Estimated"
            print(f"{i:<3} {m.get('option','?'):<4} {prov:<12} {mid:<36} {speed_val:>11.1f} t/s {src:<10} {m.get('output_limit',0):>10,}")

    def assess_input_size(
        self,
        prompt: str,
        sources: Any = None,
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt_combined = f"{system_prompt or ''}\n{prompt}".strip()
        prompt_tokens = _estimate_text_tokens(prompt_combined)
        src_stats = _load_sources_payload(sources, router=self)
        extra_imgs = ([image_path] if image_path else []) + list(image_paths or [])
        image_tokens = 1024 * len(extra_imgs)
        total_tokens = prompt_tokens + src_stats["tokens"] + image_tokens
        return {
            "chars": len(prompt_combined) + src_stats["chars"],
            "words": len(prompt_combined.split()) + src_stats["words"],
            "prompt_tokens": prompt_tokens,
            "source_tokens": src_stats["tokens"],
            "source_files": src_stats["file_count"],
            "source_kb": round(src_stats["total_bytes"] / 1024.0, 1),
            "source_summary": src_stats["summary"],
            "source_excerpt": src_stats["sample_excerpt"],
            "image_paths": list(dict.fromkeys(extra_imgs + list(src_stats.get("image_paths") or []))),
            "input_tokens": total_tokens,
        }

    def estimate_output_size_via_fast_reroute(self, prompt: str, input_tokens: int, source_info: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        heuristic = _heuristic_output_estimate(prompt, input_tokens)
        fast_candidates = self._get_speed_ranked_models(top_n=10, vision_only=False)
        if not fast_candidates:
            heuristic["assessed_by"] = "heuristic"
            return heuristic

        if len(prompt) > 2200:
            prompt_preview = f"{prompt[:1300]}\n...[truncated {len(prompt) - 2200} chars]...\n{prompt[-900:]}"
        else:
            prompt_preview = prompt

        src_tokens = (source_info or {}).get("source_tokens", 0)
        src_files = (source_info or {}).get("source_files", 0)
        src_kb = (source_info or {}).get("source_kb", 0.0)
        src_excerpt = (source_info or {}).get("source_excerpt", "")
        prompt_toks = (source_info or {}).get("prompt_tokens", input_tokens)

        sys_note = f"System role summary: {system_prompt[:250]}\n" if system_prompt else ""
        source_block = (
            f"INPUT 1 - SOURCE PAYLOAD: {src_files} file(s), ~{src_tokens:,} tokens ({src_kb:.1f} KB).\n"
            + (f"Source sample excerpt:\n\"\"\"\n{src_excerpt[:900]}\n\"\"\"\n" if src_excerpt else "")
        )
        estimator_prompt = (
            "You are sizing the output token budget when ALL provided sources together are processed through the task prompt below.\n"
            f"{source_block}"
            f"INPUT 2 - TASK PROMPT (~{prompt_toks:,} tokens):\n"
            f"{sys_note}"
            f"\"\"\"\n{prompt_preview}\n\"\"\"\n"
            f"TOTAL COMBINED INPUT TO BE SENT TOGETHER: ~{input_tokens:,} tokens.\n\n"
            "Calculate the expected output token size if all of these sources together are put through this prompt (e.g. a zero-loss union-merge retains nearly all unique source content, whereas a short summary compresses it).\n"
            "Reply ONLY with valid JSON in this exact schema:\n"
            '{"estimated_output_tokens": <integer between 64 and 65536>, "reasoning_depth": "<light|moderate|deep>", "rationale": "<max 14 words>"}'
        )

        for m in fast_candidates:
            prov, mid = m["provider"], m["id"]
            if not _model_fits_budget(m, _estimate_text_tokens(estimator_prompt) + 64, 220):
                continue
            try:
                raw = self._call_model(prov, mid, estimator_prompt, None, None, 0.0, 180)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    est_out = int(parsed.get("estimated_output_tokens") or heuristic["estimated_output_tokens"])
                    est_out = max(64, min(est_out, 65536))
                    depth = str(parsed.get("reasoning_depth") or heuristic["reasoning_depth"]).lower()
                    if depth not in ("light", "moderate", "deep"):
                        depth = heuristic["reasoning_depth"]
                    rationale = str(parsed.get("rationale") or heuristic["rationale"]).strip()
                    return {
                        "estimated_output_tokens": est_out,
                        "reasoning_depth": depth,
                        "rationale": rationale,
                        "assessed_by": f"{prov}:{mid}",
                    }
            except Exception:
                continue

        heuristic["assessed_by"] = "heuristic-fallback"
        return heuristic

    def recommend_models_for_task(
        self,
        prompt: str,
        sources: Any = None,
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        explicit_max_tokens: Optional[int] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        in_stats = self.assess_input_size(
            prompt,
            sources=sources,
            system_prompt=system_prompt,
            image_path=image_path,
            image_paths=image_paths,
        )
        vision_only = bool(in_stats.get("image_paths"))
        req_in = in_stats["input_tokens"]
        out_stats = self.estimate_output_size_via_fast_reroute(prompt, req_in, source_info=in_stats, system_prompt=system_prompt)
        req_out = max(int(out_stats["estimated_output_tokens"]), int(explicit_max_tokens or 0))

        safe_in = int(req_in * 1.15) + 64
        safe_out = int(req_out * 1.15) + 64

        eligible: List[Dict[str, Any]] = []
        total_discovered = 0
        for opt_key, items in self.models_by_option.items():
            for m in items:
                if not m.get("chat", True):
                    continue
                total_discovered += 1
                if vision_only and not m.get("vision"):
                    continue
                if not _model_fits_budget(m, safe_in, safe_out):
                    continue
                entry = dict(m)
                entry["option"] = opt_key
                entry["eff_tps"] = float(entry.get("measured_tps") or entry.get("est_tps") or _estimate_model_tps(entry.get("provider", ""), entry.get("id", "")))
                entry["total_capacity"] = int(entry.get("input_limit", 0)) + int(entry.get("output_limit", 0))
                eligible.append(entry)

        if out_stats["reasoning_depth"] == "deep":
            ranked_eligible = sorted(
                eligible,
                key=lambda m: (not _is_deep_reasoning_model(m), -m["total_capacity"], -m["eff_tps"]),
            )
        else:
            ranked_eligible = sorted(
                eligible,
                key=lambda m: (-m["eff_tps"], m["total_capacity"]),
            )

        speed_space_pick = ranked_eligible[0] if ranked_eligible else None
        deep_candidates = [m for m in ranked_eligible if _is_deep_reasoning_model(m)]
        depth_speed_pick = deep_candidates[0] if deep_candidates else speed_space_pick

        plan = {
            "prompt_tokens": in_stats["prompt_tokens"],
            "source_tokens": in_stats["source_tokens"],
            "source_files": in_stats["source_files"],
            "input_tokens": req_in,
            "estimated_output_tokens": req_out,
            "safe_input_tokens": safe_in,
            "safe_output_tokens": safe_out,
            "reasoning_depth": out_stats["reasoning_depth"],
            "rationale": out_stats["rationale"],
            "speed_space_pick": speed_space_pick,
            "depth_speed_pick": depth_speed_pick,
            "ranked_candidates": ranked_eligible,
        }
        self.eligible_models = ranked_eligible
        self.last_task_plan = plan

        if not quiet:
            print("\n=== Option 10: Smart Task-Sized Model Assessment ===")
            print(f"  Input 1 (Sources) : ~{in_stats['source_tokens']:,} tokens across {in_stats['source_summary']} ({in_stats['source_kb']:.1f} KB)")
            print(f"  Input 2 (Prompt)  : ~{in_stats['prompt_tokens']:,} tokens")
            print(f"  Combined Input    : ~{req_in:,} tokens ({in_stats['words']:,} words, {in_stats['chars']:,} chars) [safe budget: {safe_in:,}]")
            print(f"  Output Estimate   : ~{req_out:,} tokens [safe budget: {safe_out:,}] | Depth: {out_stats['reasoning_depth'].upper()} (via {out_stats['assessed_by']})")
            print(f"  Combined Analysis : {out_stats['rationale']}")
            print(f"  Eligible Pool     : {len(ranked_eligible)}/{total_discovered} cached models fit without input/output overflow")
            if speed_space_pick:
                print(f"  * Speed & Space Optimized (Rank 1) : {speed_space_pick['provider']}:{speed_space_pick['id']} (~{speed_space_pick['eff_tps']:.0f} tok/s, in={speed_space_pick['input_limit']:,}, out={speed_space_pick['output_limit']:,})")
            if depth_speed_pick:
                print(f"  * Depth & Speed Optimized          : {depth_speed_pick['provider']}:{depth_speed_pick['id']} (~{depth_speed_pick['eff_tps']:.0f} tok/s, in={depth_speed_pick['input_limit']:,}, out={depth_speed_pick['output_limit']:,})")
            if ranked_eligible:
                print("\n--- Sufficient Models Ranked by Speed (Fastest Sufficient -> Bigger Models 1 by 1) ---")
                header = f"{'Rank':<5} {'Opt':<4} {'Provider':<12} {'Model ID':<36} {'Speed':>11} {'In Limit':>10} {'Out Limit':>10}"
                print(header)
                print("-" * len(header))
                for idx, m in enumerate(ranked_eligible, 1):
                    print(f"{idx:<5} {m.get('option','?'):<4} {m['provider']:<12} {m['id'][:35]:<36} {m['eff_tps']:>7.0f} t/s {m['input_limit']:>10,} {m['output_limit']:>10,}")

        return plan

    def _read_text_or_file_arg(self, label: str, allow_dir: bool = False, return_path_only: bool = False) -> str:
        raw = input(label).strip()
        if not raw:
            return ""
        if raw.lower() == "paste":
            print("Paste your multi-line text below, then type END on a new line:")
            lines: List[str] = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            return "\n".join(lines).strip()
        if return_path_only:
            return raw
        cand = Path(raw[1:] if raw.startswith("@") else raw).expanduser()
        if allow_dir and cand.is_dir():
            return str(cand)
        if cand.is_file():
            content = cand.read_text(encoding="utf-8", errors="replace")
            print(f"[Option 10] Loaded {len(content):,} chars from {cand}")
            return content
        return raw

    def _read_trial_inputs(self) -> tuple[str, str]:
        print("\n[Option 10] Two-Input Task Sizing (Existing cached models remain untouched):")
        print("  Input 1 (Sources): Folder path (e.g. topic_clusters/04_Stereochemistry), comma-separated file paths, 'paste', or Enter if none")
        sources_arg = self._read_text_or_file_arg("  [1/2] Sources folder/files: ", allow_dir=True)
        print("  Input 2 (Prompt) : Prompt .txt/.md file path (e.g. prompt.txt), 'paste' (end with END), single-line prompt, or Enter to close Option 10")
        prompt_arg = self._read_text_or_file_arg("  [2/2] Task prompt: ", allow_dir=False)
        return sources_arg, prompt_arg

    def _prompt_custom_model(self) -> bool:
        print("\n=== Option 9: Configure Custom Model ===")
        print("Supported providers: ollama, gemini, groq, openrouter, nvidia, huggingface")
        raw = input("Enter custom model as 'provider:model_id' (or press Enter to choose provider separately, 'b' to go back): ").strip()
        if raw.lower() == "b":
            return False
        if ":" in raw and raw.split(":", 1)[0].lower() in PROVIDER_RATE_INFO:
            provider, model_id = raw.split(":", 1)
            provider, model_id = provider.strip().lower(), model_id.strip()
        else:
            provider = input("Enter provider (ollama/gemini/groq/openrouter/nvidia/huggingface): ").strip().lower()
            if provider not in PROVIDER_RATE_INFO:
                print(f"[Router] Invalid provider '{provider}'. Returning to menu.")
                return False
            model_id = raw if raw else input("Enter exact model ID: ").strip()
        if not model_id:
            print("[Router] Model ID cannot be empty. Returning to menu.")
            return False
        self.custom_model = {"provider": provider, "id": model_id}
        self.selected_option = "9"
        print(f"[Router] Active pipeline: Custom Model ({provider}:{model_id}) — will route back to choice menu on any error.")
        return True

    def configure_non_interactive(self, option: str, custom_model_spec: Optional[str] = None) -> None:
        opt = str(option).strip().lower()
        if opt == "9" or custom_model_spec:
            spec = custom_model_spec or opt
            if ":" in spec:
                prov, mid = spec.split(":", 1)
                prov, mid = prov.strip().lower(), mid.strip()
                if prov in PROVIDER_RATE_INFO:
                    self.custom_model = {"provider": prov, "id": mid}
                    self.selected_option = "9"
                    print(f"[Router] Non-interactive custom model configured: {prov}:{mid}")
                    return
            sys.exit("Error: Option 9 requires format 'provider:model_id' (e.g., groq:llama-3.3-70b-versatile).")

        if opt in self.OPTIONS or opt in ("7", "8", "10", "11", "deep"):
            self.selected_option = "deep" if opt in ("11", "deep") else opt
            print(f"[Router] Non-interactive pipeline selected: Option {self.selected_option}")
            return

        sys.exit(f"Error: Invalid option '{option}' specified. Choose from 1-10.")

    def select_provider_interactive(self, default_option: Optional[str] = None, vision_only: bool = False, preset_sources: Any = None) -> str:
        if default_option in self.OPTIONS or default_option in ("7", "8", "9", "10"):
            self.selected_option = default_option
            return default_option
        while True:
            print("\n=== Select Free LLM API Pipeline ===")
            for key, label in self.OPTIONS.items():
                count = len(self._get_option_models(key, vision_only=vision_only))
                print(f"  {key}. {label} ({count} {'vision ' if vision_only else ''}models)")
            ranked_count = len(self._get_ranked_models(top_n=10, vision_only=vision_only))
            speed_count = len(self._get_speed_ranked_models(top_n=10, vision_only=vision_only))
            print("  6. Compare all models side-by-side by Token Limits (Top 10)")
            print(f"  7. Use Top-10 Token-Ranked Models (Rank 1 -> Rank {ranked_count} fallback)")
            print(f"  8. Compare & Rank Models by Output Speed (Top {speed_count} fastest tok/s fallback)")
            print("  9. Input a Custom Model (routes back to this menu if the model throws an error)")
            print(" 10. Smart Task-Sized Optimizer (Filter existing models by input/output -> fastest first)")
            print("  r. Refresh free model lists & run Reconfiguration Drive")
            print("  c. View / re-run Reconfiguration Drive on current cached models")
            choice = input("Enter option (1-10, r, or c): ").strip().lower()
            if choice == "r":
                self.discover_and_cache_models()
                continue
            if choice == "c":
                self.run_reconfiguration_drive()
                continue
            if choice == "6":
                self.compare_top_models(top_n=10, vision_only=vision_only)
                continue
            if choice == "7":
                if not ranked_count:
                    print("[Router] No ranked models available. Press 'r' to refresh.")
                    continue
                self.selected_option = "7"
                print(f"[Router] Active pipeline: Top-10 Token-Ranked Models (Rank 1 -> Rank {ranked_count})")
                return "7"
            if choice == "8":
                if not speed_count:
                    print("[Router] No models cached yet. Press 'r' to refresh.")
                    continue
                self.compare_models_by_speed(top_n=10, vision_only=vision_only)
                sub = input("\n[Option 8] Enter 'u' to use Speed-Ranked Top-10 pipeline, 'b' to run live speed benchmark, or Enter to return to menu: ").strip().lower()
                if sub == "b":
                    self.benchmark_models_speed(top_n=10, vision_only=vision_only)
                    self.compare_models_by_speed(top_n=10, vision_only=vision_only)
                    use_now = input("Use this Speed-Ranked Top-10 pipeline now? (y/n): ").strip().lower()
                    if use_now == "y":
                        self.selected_option = "8"
                        print(f"[Router] Active pipeline: Top-10 Speed-Ranked Models (Fastest 1 -> {speed_count} fallback)")
                        return "8"
                elif sub == "u":
                    self.selected_option = "8"
                    print(f"[Router] Active pipeline: Top-10 Speed-Ranked Models (Fastest 1 -> {speed_count} fallback)")
                    return "8"
                continue
            if choice == "9":
                if self._prompt_custom_model():
                    return "9"
                continue
            if choice == "10":
                if not speed_count:
                    print("[Router] No models cached yet. Press 'r' to refresh.")
                    continue
                if preset_sources is not None:
                    print("[Option 10] Preloaded chatbot sources detected — models will be auto-sized on every prompt.")
                    sample_prompt = self._read_text_or_file_arg("  Optional preview prompt (or press Enter to activate Option 10 now): ", allow_dir=False)
                    if sample_prompt:
                        self.recommend_models_for_task(sample_prompt, sources=preset_sources)
                else:
                    trial_sources, trial_prompt = self._read_trial_inputs()
                    if trial_sources or trial_prompt:
                        self.recommend_models_for_task(trial_prompt, sources=trial_sources)
                        print(f"[Option 10] Saved {len(self.eligible_models)} eligible models (full cache untouched). Closing Option 10...")
                self.selected_option = "10"
                print("[Router] Active pipeline: Option 10 Smart Task-Sized Optimizer (Fastest sufficient model first)")
                return "10"

            if choice in self.OPTIONS:
                if not self._get_option_models(choice, vision_only=vision_only):
                    print(f"[Router] No models cached for option {choice}. Press 'r' to refresh.")
                    continue
                self.selected_option = choice
                print(f"[Router] Active pipeline: {self.OPTIONS[choice]}")
                return choice
            print("Invalid selection. Enter 1-10 or r.")

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        sources: Any = None,
        image_paths: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        validator: Optional[Any] = None,
    ) -> tuple[str, str]:
        all_images = list(dict.fromkeys(([image_path] if image_path else []) + list(image_paths or [])))
        vision_only = bool(all_images)
        while True:
            if not self.selected_option:
                if not sys.stdin.isatty():
                    auto_opt = "10" if (sources is not None or all_images) else "8"
                    self.configure_non_interactive(auto_opt)
                else:
                    self.select_provider_interactive(vision_only=vision_only)

            if self.selected_option == "9":
                if not self.custom_model:
                    if not sys.stdin.isatty():
                        sys.exit("Error: Custom model configuration required in non-interactive mode.")
                    self.selected_option = None
                    continue
                provider = self.custom_model["provider"]
                mid = self.custom_model["id"]
                print(f"[Router] Trying Custom Model {provider}:{mid} ...")
                try:
                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, max_tokens, image_paths=all_images, history=history)
                    return text, f"{provider}:{mid}"
                except Exception as exc:
                    safe_exc = _redact_secrets(exc)
                    print(f"\n[Custom Model Error] {provider}:{mid} threw an error: {safe_exc}")
                    if not sys.stdin.isatty():
                        sys.exit(f"Error in non-interactive mode: Custom model {provider}:{mid} failed: {safe_exc}")
                    print("[Router] Routing you back to the choice options menu to select an alternative pipeline...")
                    self.selected_option = None
                    self.custom_model = None
                    continue

            task_plan: Optional[Dict[str, Any]] = None
            if self.selected_option == "7":
                candidates = self._get_ranked_models(top_n=10, vision_only=vision_only)
            elif self.selected_option == "8":
                candidates = self._get_speed_ranked_models(top_n=10, vision_only=vision_only)
            elif self.selected_option == "deep":
                lookup = {(m["provider"], m["id"]): m for m in self._all_discovered_models() if m.get("chat", True) and (not vision_only or m.get("vision"))}
                candidates = [lookup[(r["provider"], r["id"])] for r in (self.role_config.get("deep_synthesis") or []) if (r["provider"], r["id"]) in lookup]
                seen = {(m["provider"], m["id"]) for m in candidates}
                candidates.extend(m for m in self._get_ranked_models(top_n=None, vision_only=vision_only) if (m["provider"], m["id"]) not in seen)
            elif self.selected_option == "10":
                history_text = "\n".join(m.get("content", "") for m in (history or []))
                sizing_prompt = f"{history_text}\n{prompt}".strip() if history_text else prompt
                task_plan = self.recommend_models_for_task(
                    sizing_prompt,
                    sources=sources,
                    system_prompt=system_prompt,
                    image_paths=all_images,
                    explicit_max_tokens=max_tokens,
                )
                candidates = task_plan["ranked_candidates"]
            else:
                candidates = self._get_option_models(self.selected_option or "", vision_only=vision_only)

            if not candidates:
                print(f"[Router] No models available for option {self.selected_option}. Routing back to choice menu...")
                self.selected_option = None
                continue

            errors = []
            now = time.time()
            active_candidates = [m for m in candidates if self._cooldown_until.get((m["provider"], m["id"]), 0.0) <= now] or candidates
            for idx, model in enumerate(active_candidates):
                mid, provider = model["id"], model["provider"]
                total = model.get("input_limit", 0) + model.get("output_limit", 0)
                tps = model.get("measured_tps") or model.get("est_tps") or _estimate_model_tps(provider, mid)
                effective_max_tokens = max_tokens
                if task_plan is not None:
                    target_out = max_tokens or task_plan["safe_output_tokens"]
                    effective_max_tokens = min(target_out, int(model.get("output_limit") or target_out))
                print(f"[Router] Trying ({idx + 1}/{len(active_candidates)}) {provider}:{mid} [total_limit={total:,}, ~{float(tps):.0f} tok/s]")
                try:
                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, effective_max_tokens, image_paths=all_images, history=history)
                    if any(text.count(text[i : i + 25]) > 8 for i in range(0, min(len(text), 4000), 25) if len(text[i : i + 25].strip()) >= 15):
                        raise ValueError("Degenerate substring repetition loop detected")
                    if validator is not None:
                        val_err = validator(text)
                        if val_err:
                            raise ValueError(f"Validator rejected output: {val_err}")
                    return text, f"{provider}:{mid}"
                except Exception as exc:
                    safe_exc = _redact_secrets(exc)
                    if "429" in safe_exc:
                        self._cooldown_until[(provider, mid)] = time.time() + 60.0
                    errors.append(f"{provider}:{mid} -> {safe_exc}")
                    print(f"[Fallback] {provider}:{mid} failed ({safe_exc}). Rerouting to next best model...")

            print("[Router] All models in the selected option failed:\n" + "\n".join(errors))
            if not sys.stdin.isatty():
                sys.exit("Error in non-interactive mode: All models in selected option failed.")
            print("[Router] Routing back to the choice options menu...")
            self.selected_option = None

    def _call_model(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        image_path: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        image_paths: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        native_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        resolved_imgs = list(dict.fromkeys(([image_path] if image_path else []) + list(image_paths or [])))
        encoded_imgs = [_encode_image(p) for p in resolved_imgs if Path(p).is_file()]
        prior_turns = list(history or [])
        self.last_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "source": "estimated"}

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
            if encoded_imgs:
                user_msg["images"] = [b64 for b64, _ in encoded_imgs]
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + prior_turns + [user_msg]
            resp = requests.post(f"{base_url}/api/chat", json={"model": model_id, "messages": msgs, "stream": False}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            if not text:
                raise ValueError("Empty response from Ollama")
            p_tok = int(data.get("prompt_eval_count") or 0)
            c_tok = int(data.get("eval_count") or 0)
            if p_tok > 0 or c_tok > 0:
                self.last_call_usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "source": "api"}
            return text

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            contents: List[Dict[str, Any]] = []
            for turn in prior_turns:
                role = "model" if turn.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})
            parts: List[Dict[str, Any]] = []
            for b64_img, mime_type in encoded_imgs:
                parts.append({"inline_data": {"mime_type": mime_type, "data": b64_img}})
            parts.append({"text": prompt})
            contents.append({"role": "user", "parts": parts})
            payload: Dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": temperature}}
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            if max_tokens:
                payload["generationConfig"]["maxOutputTokens"] = max_tokens
            if native_tools and "gemma" not in model_id.lower():
                from broutegpt_agent.tool_schemas import to_gemini_tools

                payload["tools"] = to_gemini_tools(native_tools)
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            cand0 = data["candidates"][0]
            out_parts = cand0.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in out_parts)
            fn_calls = [
                {"name": p["functionCall"].get("name", ""), "args": p["functionCall"].get("args") or {}}
                for p in out_parts
                if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)
            ]
            if fn_calls:
                from broutegpt_agent.tool_schemas import normalize_native_tool_calls

                text = normalize_native_tool_calls(fn_calls, thought_text=text)
            if not text:
                raise ValueError("Empty Gemini response")
            usage_meta = data.get("usageMetadata") or {}
            p_tok = int(usage_meta.get("promptTokenCount") or 0)
            c_tok = int(usage_meta.get("candidatesTokenCount") or 0)
            if p_tok > 0 or c_tok > 0:
                self.last_call_usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "source": "api"}
            if cand0.get("finishReason") == "MAX_TOKENS" and text.count("```") % 2 == 1:
                raise ValueError("Gemini output truncated mid-code-block (MAX_TOKENS)")
            return text

        endpoints = {
            "groq": ("[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)", "GROQ_API_KEY"),
            "openrouter": ("[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)", "OPENROUTER_API_KEY"),
            "nvidia": ("[https://integrate.api.nvidia.com/v1/chat/completions](https://integrate.api.nvidia.com/v1/chat/completions)", "NVIDIA_API_KEY"),
            "huggingface": ("[https://router.huggingface.co/v1/chat/completions](https://router.huggingface.co/v1/chat/completions)", "HF_API_KEY"),
        }
        url, env_var = endpoints[provider]
        if encoded_imgs:
            user_content: Any = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
                for b64_img, mime_type in encoded_imgs
            ]
        else:
            user_content = prompt
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + prior_turns + [{"role": "user", "content": user_content}]
        payload: Dict[str, Any] = {"model": model_id, "messages": msgs, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if native_tools:
            from broutegpt_agent.tool_schemas import to_openai_tools

            payload["tools"] = to_openai_tools(native_tools)
            payload["tool_choice"] = "auto"
        resp = requests.post(url, headers={"Authorization": f"Bearer {os.getenv(env_var)}", "Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        choice0 = data["choices"][0]
        msg0 = choice0.get("message") or {}
        text = msg0.get("content") or ""
        raw_tcalls = msg0.get("tool_calls") or []
        if raw_tcalls:
            from broutegpt_agent.tool_schemas import normalize_native_tool_calls

            parsed_calls = []
            for tc in raw_tcalls:
                fn_obj = tc.get("function") or {}
                parsed_calls.append({
                    "name": fn_obj.get("name", ""),
                    "args": fn_obj.get("arguments") or {},
                })
            text = normalize_native_tool_calls(parsed_calls, thought_text=text)
        if not text:
            raise ValueError("Empty response")
        usage_obj = data.get("usage") or {}
        p_tok = int(usage_obj.get("prompt_tokens") or 0)
        c_tok = int(usage_obj.get("completion_tokens") or 0)
        if p_tok > 0 or c_tok > 0:
            self.last_call_usage = {"prompt_tokens": p_tok, "completion_tokens": c_tok, "source": "api"}
        if choice0.get("finish_reason") == "length" and text.count("```") % 2 == 1:
            raise ValueError("Output truncated mid-code-block (finish_reason=length)")
        return text

