import argparse
import base64
import datetime
import html as html_lib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _redact_secrets(text: Any) -> str:
    return re.sub(
        r"([?&](?:key|api_key|apikey|token)=)[^&\s'\"]+",
        r"\1REDACTED",
        str(text),
        flags=re.IGNORECASE,
    )


def _load_env_file(env_path: str = ".env") -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_or_prompt_api_key(key_name: str, display_name: str, required: bool = False) -> Optional[str]:
    api_key = os.environ.get(key_name)
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        return api_key.strip()
    if not required:
        return None
    if not sys.stdin.isatty():
        print(f"Error: {display_name} not found in environment and running non-interactively.")
        sys.exit(1)
    print(f"Warning: {display_name} not found in .env.")
    user_input = input(f"Please enter your {display_name}: ").strip()
    if not user_input:
        sys.exit(f"Error: {display_name} cannot be empty.")
    return user_input


def load_all_api_keys(env_path: str = ".env") -> Dict[str, str]:
    _load_env_file(env_path)
    loaded = {}
    for key_name, label in [
        ("GEMINI_API_KEY", "Gemini API Key"),
        ("GROQ_API_KEY", "Groq API Key"),
        ("OPENROUTER_API_KEY", "OpenRouter API Key"),
        ("NVIDIA_API_KEY", "NVIDIA API Key"),
        ("HF_API_KEY", "Hugging Face API Key"),
    ]:
        val = get_or_prompt_api_key(key_name, label, required=False)
        if val:
            loaded[key_name] = val
    return loaded


DEFAULT_CACHE_FILE = ".free_models_cache.json"

PROVIDER_RATE_INFO = {
    "ollama": ("Unlimited (Local)", "100% Free (Local)"),
    "gemini": ("~300-900 RPH (Flash)", "Free tier (Pro=0 quota)"),
    "groq": ("~1,800 RPH (30 RPM)", "Free tier (RPD capped)"),
    "openrouter": ("~1,200 RPH (50 RPD)", "100% Free (:free)"),
    "nvidia": ("~2,400 RPH (40 RPM)", "Free dev credits"),
    "huggingface": ("~100-300 RPH", "Free monthly quota"),
}


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


def _is_valid_key(val: Optional[str]) -> bool:
    return bool(val and val.strip() and not val.strip().startswith("your_"))


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


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    char_est = int(len(text) / 3.8)
    word_est = int(len(text.split()) * 1.35)
    return max(char_est, word_est, 1)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _extract_zip_xml_text(fpath: Path, xml_prefixes: tuple[str, ...]) -> str:
    chunks: List[str] = []
    with zipfile.ZipFile(fpath, "r") as zf:
        members = sorted(
            n for n in zf.namelist()
            if any(n.startswith(pref) for pref in xml_prefixes) and n.endswith(".xml")
        )
        for member in members:
            try:
                root = ET.fromstring(zf.read(member))
                texts = [el.text.strip() for el in root.iter() if el.text and el.text.strip()]
                if texts:
                    chunks.append(" ".join(texts))
            except Exception:
                continue
    return "\n\n".join(chunks)


def _transcribe_media_file(fpath: Path, router: Optional["FreeLLMRouter"] = None) -> str:
    audio_candidates: List[Dict[str, str]] = []
    if router is not None:
        audio_candidates = list(router.role_config.get("audio_transcription") or [])

    if not audio_candidates:
        if _is_valid_key(os.getenv("GROQ_API_KEY")):
            audio_candidates.append({"provider": "groq", "id": "whisper-large-v3-turbo"})
        if _is_valid_key(os.getenv("HF_API_KEY")):
            audio_candidates.append({"provider": "huggingface", "id": "openai/whisper-large-v3-turbo"})

    mime_type, _ = mimetypes.guess_type(str(fpath))
    for cand in audio_candidates:
        prov = cand.get("provider", "")
        mid = cand.get("id", "")
        if prov == "groq" and _is_valid_key(os.getenv("GROQ_API_KEY")):
            try:
                with fpath.open("rb") as fh:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                        files={"file": (fpath.name, fh, mime_type or "application/octet-stream")},
                        data={"model": mid},
                        timeout=120,
                    )
                if resp.ok:
                    transcript = resp.json().get("text", "").strip()
                    if transcript:
                        return f"[Audio/Video Transcript of {fpath.name} via {prov}:{mid}]\n{transcript}"
            except Exception as exc:
                print(f"[Sources] Audio transcription warning ({prov}:{mid}) for {fpath.name}: {_redact_secrets(exc)}")
        elif prov == "huggingface" and _is_valid_key(os.getenv("HF_API_KEY")):
            try:
                raw_bytes = fpath.read_bytes()
                resp = requests.post(
                    f"https://router.huggingface.co/hf-inference/models/{mid}",
                    headers={
                        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
                        "Content-Type": mime_type or "application/octet-stream",
                    },
                    data=raw_bytes,
                    timeout=120,
                )
                if resp.ok:
                    transcript = resp.json().get("text", "").strip()
                    if transcript:
                        return f"[Audio/Video Transcript of {fpath.name} via {prov}:{mid}]\n{transcript}"
            except Exception as exc:
                print(f"[Sources] Audio transcription warning ({prov}:{mid}) for {fpath.name}: {_redact_secrets(exc)}")

    return f"[Media file attached: {fpath.name} ({fpath.stat().st_size:,} bytes) - configure GROQ_API_KEY or HF_API_KEY in .env and refresh models for automatic speech transcription]"


def _extract_file_source(
    fpath: Path,
    router: Optional["FreeLLMRouter"] = None,
    transcribe_images: bool = False,
) -> tuple[str, List[str]]:
    ext = fpath.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        resolved = str(fpath.resolve())
        if transcribe_images and router is not None:
            transcript, used = router.transcribe_image(resolved)
            return f"[Image Transcription of {fpath.name} via {used}]\n{transcript}", [resolved]
        return f"[Attached Image: {fpath.name}]", [resolved]

    if ext in MEDIA_EXTENSIONS:
        return _transcribe_media_file(fpath, router=router), []

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(fpath))
            pages_text = [f"--- Page {i + 1} ---\n{page.get_text('text').strip()}" for i, page in enumerate(doc)]
            combined = "\n\n".join(p for p in pages_text if p.strip())
            if len(combined.strip()) >= 80:
                doc.close()
                return combined, []
            tmp_dir = Path(tempfile.mkdtemp(prefix="router_pdf_"))
            rendered: List[str] = []
            transcribed_pages: List[str] = []
            for i in range(min(len(doc), 5)):
                out_img = tmp_dir / f"{fpath.stem}_p{i + 1}.png"
                doc[i].get_pixmap(dpi=150).save(str(out_img))
                rendered.append(str(out_img))
                if transcribe_images and router is not None:
                    page_txt, _ = router.transcribe_image(str(out_img))
                    transcribed_pages.append(f"--- Scanned Page {i + 1} ---\n{page_txt}")
            doc.close()
            extra_txt = "\n\n".join(transcribed_pages)
            return f"[Scanned PDF '{fpath.name}': {len(rendered)} page(s)]\n{combined}\n{extra_txt}".strip(), rendered
        except ImportError:
            pass
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(fpath))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages), []
        except Exception as exc:
            return f"[Could not extract PDF {fpath.name}: install pymupdf or pypdf ({exc})]", []

    if ext == ".docx":
        try:
            import docx

            doc = docx.Document(str(fpath))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()), []
        except Exception:
            return _extract_zip_xml_text(fpath, ("word/document",)), []

    if ext == ".pptx":
        try:
            from pptx import Presentation

            prs = Presentation(str(fpath))
            slides = []
            for idx, slide in enumerate(prs.slides, 1):
                lines = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
                if lines:
                    slides.append(f"--- Slide {idx} ---\n" + "\n".join(lines))
            return "\n\n".join(slides), []
        except Exception:
            return _extract_zip_xml_text(fpath, ("ppt/slides/slide",)), []

    if ext == ".xlsx":
        try:
            import openpyxl

            wb = openpyxl.load_workbook(str(fpath), data_only=True)
            sheets = []
            for name in wb.sheetnames:
                ws = wb[name]
                rows = ["\t".join("" if v is None else str(v) for v in row) for row in ws.iter_rows(values_only=True)]
                sheets.append(f"--- Sheet: {name} ---\n" + "\n".join(r for r in rows if r.strip()))
            return "\n\n".join(sheets), []
        except Exception:
            return _extract_zip_xml_text(fpath, ("xl/sharedStrings", "xl/worksheets/sheet")), []

    return fpath.read_text(encoding="utf-8", errors="replace"), []


def _fetch_url_source(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        body = resp.text
        body = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", body)
        body = re.sub(r"(?s)<[^>]+>", " ", body)
        body = html_lib.unescape(re.sub(r"\s+", " ", body)).strip()
        return body[:50000]
    except Exception as exc:
        return f"[Failed to fetch URL {_redact_secrets(url)}: {_redact_secrets(exc)}]"


def _load_sources_payload(
    sources: Any,
    router: Optional["FreeLLMRouter"] = None,
    transcribe_images: bool = False,
) -> Dict[str, Any]:
    if isinstance(sources, dict) and "full_text" in sources and "tokens" in sources:
        return sources
    if not sources:
        return {
            "file_count": 0,
            "total_bytes": 0,
            "chars": 0,
            "words": 0,
            "tokens": 0,
            "summary": "None",
            "sample_excerpt": "",
            "full_text": "",
            "image_paths": [],
        }

    items: List[Any] = []

    def _safe_expand_path(path_str: str) -> Path:
        p = Path(path_str)
        if path_str == "~" or path_str.startswith("~/"):
            try:
                return p.expanduser()
            except RuntimeError:
                return Path(os.path.expanduser(path_str))
        return p

    if isinstance(sources, (str, Path)):
        raw_str = str(sources).strip()
        if "\n" not in raw_str and "," in raw_str and not _safe_expand_path(raw_str).exists():
            items = [p.strip() for p in raw_str.split(",") if p.strip()]
        else:
            items = [raw_str]
    elif isinstance(sources, (list, tuple, set)):
        items = list(sources)
    else:
        items = [str(sources)]

    file_count = 0
    total_bytes = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    names: List[str] = []
    excerpts: List[str] = []
    full_blocks: List[str] = []
    image_paths: List[str] = []

    def _record_entry(label: str, txt: str, size_bytes: int, imgs: List[str]) -> None:
        nonlocal file_count, total_bytes, total_chars, total_words, total_tokens
        file_count += 1
        total_bytes += size_bytes
        total_chars += len(txt)
        total_words += len(txt.split())
        total_tokens += _estimate_text_tokens(txt) + (1024 * len(imgs))
        names.append(label)
        image_paths.extend(imgs)
        if txt.strip():
            full_blocks.append(f"=== SOURCE: {label} ===\n{txt.strip()}")
            if len(excerpts) < 2:
                excerpts.append(f"[{label}]: {txt.strip()[:500]}")

    for item in items:
        if isinstance(item, (str, Path)):
            s = str(item).strip().strip("'\"")
            if not s:
                continue
            if s.startswith(("http://", "https://")):
                txt = _fetch_url_source(s)
                _record_entry(s, txt, len(txt.encode("utf-8")), [])
                continue
            if "\n" not in s and len(s) < 1024:
                raw_path_str = s[1:] if s.startswith("@") else s
                doubled_match = re.search(r"(/Users/[^/]+/.*)$", raw_path_str)
                if doubled_match and not _safe_expand_path(raw_path_str).exists():
                    repaired = Path(doubled_match.group(1))
                    if repaired.exists():
                        print(f"[Sources] Auto-repaired doubled path -> {repaired}")
                        raw_path_str = str(repaired)

                cand = _safe_expand_path(raw_path_str)
                if cand.is_dir():
                    files = sorted(
                        p for p in cand.rglob("*")
                        if p.is_file() and not any(part.startswith(".") for part in p.parts)
                    )
                    for fpath in files:
                        try:
                            txt, imgs = _extract_file_source(fpath, router=router, transcribe_images=transcribe_images)
                        except Exception:
                            continue
                        _record_entry(fpath.name, txt, fpath.stat().st_size, imgs)
                    continue
                if cand.is_file():
                    txt, imgs = _extract_file_source(cand, router=router, transcribe_images=transcribe_images)
                    _record_entry(cand.name, txt, cand.stat().st_size, imgs)
                    continue

        raw_txt = str(item).strip()
        looks_like_path = (
            len(raw_txt.splitlines()) == 1
            and len(raw_txt) < 1024
            and (
                raw_txt.startswith(("/", "~/", "./", "../", "@"))
                or re.search(r"\.(pdf|docx|pptx|xlsx|txt|md|py|json|csv|png|jpg|jpeg|webp|mp3|mp4|wav|m4a)$", raw_txt, re.IGNORECASE)
            )
        )
        if looks_like_path:
            print(f"[Sources Warning] File or folder not found (skipped): {raw_txt}")
            continue
        _record_entry(f"inline_source_{file_count + 1}", raw_txt, len(raw_txt.encode("utf-8")), [])

    preview_names = ", ".join(names[:4]) + (f" (+{len(names) - 4} more)" if len(names) > 4 else "")
    summary = f"{file_count} source(s) [{preview_names}]" if file_count else "None"
    if image_paths:
        summary += f" | {len(image_paths)} image(s)"
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "chars": total_chars,
        "words": total_words,
        "tokens": total_tokens,
        "summary": summary,
        "sample_excerpt": "\n".join(excerpts),
        "full_text": "\n\n".join(full_blocks),
        "image_paths": image_paths,
    }


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


def _is_deep_reasoning_model(model: Dict[str, Any]) -> bool:
    mid = model.get("id", "").lower()
    tokens = set(re.split(r"[-_/:]+", mid))
    if tokens & {"0.5b", "1b", "1.5b", "2b", "3b", "4b", "7b", "8b", "9b", "12b", "lite", "mini", "nano", "tiny"}:
        return False
    if "r1" in tokens:
        return True
    return any(k in mid for k in ("70b", "72b", "120b", "235b", "405b", "qwen3", "qwq", "versatile", "llama-3.3", "llama-4", "maverick", "flash", "27b", "large", "plus"))


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

    def __init__(self, env_path: str = ".env", cache_path: str = DEFAULT_CACHE_FILE, force_refresh: bool = False, timeout: int = 90) -> None:
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
        deep_models = [m for m in cap_sorted if _is_deep_reasoning_model(m)] or cap_sorted

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
    ) -> str:
        resolved_imgs = list(dict.fromkeys(([image_path] if image_path else []) + list(image_paths or [])))
        encoded_imgs = [_encode_image(p) for p in resolved_imgs if Path(p).is_file()]
        prior_turns = list(history or [])

        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
            if encoded_imgs:
                user_msg["images"] = [b64 for b64, _ in encoded_imgs]
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + prior_turns + [user_msg]
            resp = requests.post(f"{base_url}/api/chat", json={"model": model_id, "messages": msgs, "stream": False}, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "")
            if not text:
                raise ValueError("Empty response from Ollama")
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
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            cand0 = resp.json()["candidates"][0]
            out_parts = cand0["content"]["parts"]
            text = "".join(p.get("text", "") for p in out_parts)
            if not text:
                raise ValueError("Empty Gemini response")
            if cand0.get("finishReason") == "MAX_TOKENS" and text.count("```") % 2 == 1:
                raise ValueError("Gemini output truncated mid-code-block (MAX_TOKENS)")
            return text

        endpoints = {
            "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
            "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
            "huggingface": ("https://router.huggingface.co/v1/chat/completions", "HF_API_KEY"),
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
        payload = {"model": model_id, "messages": msgs, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        resp = requests.post(url, headers={"Authorization": f"Bearer {os.getenv(env_var)}", "Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        choice0 = resp.json()["choices"][0]
        text = choice0["message"]["content"]
        if not text:
            raise ValueError("Empty response")
        if choice0.get("finish_reason") == "length" and text.count("```") % 2 == 1:
            raise ValueError("Output truncated mid-code-block (finish_reason=length)")
        return text


# ============================================================================
# STANDALONE CHATBOT MODE & MULTI-FORMAT OUTPUT ENGINE
# (Only runs when executing `python3 broutegpt.py` directly;
#  importing FreeLLMRouter from scripts like llm_connector.py stays unaffected.)
# ============================================================================

SUPPORTED_OUTPUT_FORMATS = ("auto", "chat", "md", "pdf", "code", "image", "videotext", "html", "docx", "json", "csv")

CODE_LANG_EXT = {
    "python": ".py", "py": ".py", "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
    "bash": ".sh", "sh": ".sh", "zsh": ".sh", "html": ".html", "css": ".css", "json": ".json",
    "csv": ".csv", "sql": ".sql", "yaml": ".yaml", "yml": ".yml", "latex": ".tex", "tex": ".tex",
    "cpp": ".cpp", "c++": ".cpp", "c": ".c", "java": ".java", "rust": ".rs", "go": ".go",
    "r": ".R", "mermaid": ".mmd", "svg": ".svg", "xml": ".xml", "markdown": ".md", "md": ".md",
}


def _detect_output_format(prompt: str, requested_format: str = "auto") -> str:
    if requested_format and requested_format != "auto":
        return requested_format
    low = prompt.lower()
    if re.search(r"\b(as|to|into|export|make|generate|create|save|output|new)\b.{0,18}\bpdf\b|\bpdf\b.{0,12}\b(file|output|export|version)\b", low):
        return "pdf"
    if any(k in low for k in ("generate an image", "create an image", "draw an image", "make an image", "generate image", "render an image", "create a picture", "draw a diagram", "make a poster")):
        return "image"
    if any(k in low for k in ("videotext", "video text", "video script", "storyboard", "srt subtitles", "narration script", "explainer video script")):
        return "videotext"
    if any(k in low for k in ("as docx", "word doc", "word document", ".docx")):
        return "docx"
    if any(k in low for k in ("as html", "html page", "interactive webpage", "save as html", ".html")):
        return "html"
    if any(k in low for k in ("as csv", "csv table", "export csv", ".csv")):
        return "csv"
    if any(k in low for k in ("as json", "return json", "export json", "valid json")):
        return "json"
    if any(k in low for k in ("as markdown", "save as md", "export md", ".md file")):
        return "md"
    if any(k in low for k in ("write code", "python script", "full script", "complete code", "save code", "write a script")):
        return "code"
    return "chat"


def _build_format_system_instruction(fmt: str) -> str:
    base = "You are a helpful, accurate, high-signal AI assistant grounded in any attached source material."
    rules = {
        "pdf": "Format your response as clean, publication-ready Markdown with clear headings, tables, and structured sections suitable for PDF export.",
        "md": "Format your response as clean, well-structured GitHub-flavored Markdown.",
        "html": "Return a complete, self-contained, modern HTML5 document inside a ```html fenced block.",
        "code": "Return complete, runnable code inside properly language-tagged fenced code blocks (e.g., ```python). Include a top comment `# filename: <name.ext>` if producing multiple files.",
        "json": "Return strictly valid JSON inside a ```json fenced block with no extra commentary.",
        "csv": "Return clean, valid RFC-4180 CSV data inside a ```csv fenced block.",
        "docx": "Format your response with clear Markdown headings, bullet lists, and paragraphs ready for DOCX document conversion.",
        "videotext": (
            "Produce a complete VideoText production package with:\n"
            "1. Scene-by-scene narration script, visual/animation cues, and on-screen text overlays.\n"
            "2. A complete timestamped subtitle file inside a ```srt fenced code block."
        ),
        "image": (
            "Write a single vivid, detailed image-generation prompt (max 90 words) based on the user's request and attached sources. "
            "Output ONLY the raw image prompt with no preamble or quotes."
        ),
    }
    extra = rules.get(fmt, "")
    return f"{base}\n{extra}".strip()


def _markdown_to_html(md_text: str, title: str = "AI Output") -> str:
    try:
        import markdown

        body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
    except ImportError:
        escaped = html_lib.escape(md_text)
        escaped = re.sub(r"(?m)^### (.+)$", r"<h3>\1</h3>", escaped)
        escaped = re.sub(r"(?m)^## (.+)$", r"<h2>\1</h2>", escaped)
        escaped = re.sub(r"(?m)^# (.+)$", r"<h1>\1</h1>", escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        body = "<p>" + escaped.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</p>"
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html_lib.escape(title)}</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6;max-width:860px;margin:32px auto;padding:0 24px;color:#1a1a1a;}"
        "h1,h2,h3{color:#111;margin-top:1.4em;} code{background:#f4f4f6;padding:2px 5px;border-radius:4px;font-size:0.92em;}"
        "pre{background:#f4f4f6;padding:12px 14px;border-radius:6px;overflow-x:auto;} table{border-collapse:collapse;width:100%;margin:16px 0;}"
        "th,td{border:1px solid #d4d4d8;padding:8px 10px;text-align:left;} th{background:#f4f4f6;}"
        "</style></head><body>"
        f"{body}</body></html>"
    )


def _render_markdown_to_pdf(md_text: str, out_path: Path) -> tuple[Path, str]:
    html_doc = _markdown_to_html(md_text, title=out_path.stem)
    try:
        from weasyprint import HTML

        HTML(string=html_doc).write_pdf(str(out_path))
        return out_path, "weasyprint"
    except Exception:
        pass

    pandoc_bin = shutil.which("pandoc")
    if pandoc_bin:
        tmp_md = out_path.with_suffix(".tmp.md")
        tmp_md.write_text(md_text, encoding="utf-8")
        try:
            res = subprocess.run([pandoc_bin, str(tmp_md), "-o", str(out_path)], capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and out_path.exists():
                return out_path, "pandoc"
        except Exception:
            pass
        finally:
            tmp_md.unlink(missing_ok=True)

    try:
        import fitz  # PyMuPDF

        story = fitz.Story(html=html_doc)
        writer = fitz.DocumentWriter(str(out_path))
        mediabox = fitz.paper_rect("a4")
        where = mediabox + (42, 42, -42, -42)
        more = 1
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        return out_path, "pymupdf"
    except Exception:
        pass

    fallback_html = out_path.with_suffix(".html")
    fallback_html.write_text(html_doc, encoding="utf-8")
    return fallback_html, "html-fallback (install weasyprint or pymupdf for direct PDF)"


def _render_markdown_to_docx(md_text: str, out_path: Path) -> tuple[Path, str]:
    try:
        import docx

        doc = docx.Document()
        for line in md_text.splitlines():
            s = line.strip()
            if s.startswith("### "):
                doc.add_heading(s[4:], level=3)
            elif s.startswith("## "):
                doc.add_heading(s[3:], level=2)
            elif s.startswith("# "):
                doc.add_heading(s[2:], level=1)
            elif s:
                doc.add_paragraph(line)
        doc.save(str(out_path))
        return out_path, "python-docx"
    except Exception:
        pass

    pandoc_bin = shutil.which("pandoc")
    if pandoc_bin:
        tmp_md = out_path.with_suffix(".tmp.md")
        tmp_md.write_text(md_text, encoding="utf-8")
        try:
            res = subprocess.run([pandoc_bin, str(tmp_md), "-o", str(out_path)], capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and out_path.exists():
                return out_path, "pandoc"
        finally:
            tmp_md.unlink(missing_ok=True)

    fallback_md = out_path.with_suffix(".md")
    fallback_md.write_text(md_text, encoding="utf-8")
    return fallback_md, "md-fallback (install python-docx or pandoc for .docx)"


def _generate_free_image(image_prompt: str, out_path: Path, router: FreeLLMRouter) -> tuple[Path, str]:
    configured_img_models = list(router.role_config.get("image_generation") or [])
    for cand in configured_img_models:
        prov = cand.get("provider", "")
        mid = cand.get("id", "")
        if prov == "huggingface" and _is_valid_key(os.getenv("HF_API_KEY")):
            try:
                resp = requests.post(
                    f"https://router.huggingface.co/hf-inference/models/{mid}",
                    headers={"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"},
                    json={"inputs": image_prompt},
                    timeout=120,
                )
                if resp.ok and resp.headers.get("content-type", "").startswith("image/"):
                    out_path.write_bytes(resp.content)
                    return out_path, f"huggingface:{mid}"
            except Exception as exc:
                print(f"[Image] Configured HF model {mid} fallback note: {_redact_secrets(exc)}")

        elif prov == "openrouter" and _is_valid_key(os.getenv("OPENROUTER_API_KEY")):
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mid,
                        "messages": [{"role": "user", "content": image_prompt}],
                        "modalities": ["image", "text"],
                    },
                    timeout=120,
                )
                if resp.ok:
                    msg = resp.json().get("choices", [{}])[0].get("message", {})
                    images = msg.get("images") or []
                    if images:
                        data_url = images[0].get("image_url", {}).get("url", "")
                        if "," in data_url:
                            out_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                            return out_path, f"openrouter:{mid}"
            except Exception as exc:
                print(f"[Image] Configured OpenRouter model {mid} fallback note: {_redact_secrets(exc)}")

        elif prov == "gemini" and _is_valid_key(os.getenv("GEMINI_API_KEY")):
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent?key={os.getenv('GEMINI_API_KEY')}",
                    json={"contents": [{"role": "user", "parts": [{"text": image_prompt}]}]},
                    timeout=120,
                )
                if resp.ok:
                    parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    for part in parts:
                        inline = part.get("inlineData") or part.get("inline_data")
                        if inline and inline.get("data"):
                            out_path.write_bytes(base64.b64decode(inline["data"]))
                            return out_path, f"gemini:{mid}"
            except Exception as exc:
                print(f"[Image] Configured Gemini model {mid} fallback note: {_redact_secrets(exc)}")

    encoded = urllib.parse.quote(image_prompt[:600])
    pol_key = os.getenv("POLLINATIONS_API_KEY")
    endpoints = []
    if _is_valid_key(pol_key):
        endpoints.append((f"https://gen.pollinations.ai/image/{encoded}", {"Authorization": f"Bearer {pol_key}"}, "pollinations:api"))
    endpoints.append((f"https://image.pollinations.ai/prompt/{encoded}?nologo=true", {}, "pollinations:legacy"))

    for url, headers, label in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=90)
            if resp.ok and resp.headers.get("content-type", "").startswith("image/") and len(resp.content) > 1000:
                out_path.write_bytes(resp.content)
                return out_path, label
        except Exception:
            continue

    # Offline / zero-quota fallback: ask active routed LLM to generate a clean SVG illustration
    svg_text, used_model = router.query(
        f"Create a clean, visually appealing standalone SVG (800x600) illustrating:\n{image_prompt}\n"
        "Return ONLY valid <svg ...>...</svg> markup.",
        temperature=0.3,
    )
    match = re.search(r"<svg[\s\S]*?</svg>", svg_text, re.IGNORECASE)
    svg_clean = match.group(0) if match else svg_text
    svg_path = out_path.with_suffix(".svg")
    svg_path.write_text(svg_clean, encoding="utf-8")
    return svg_path, f"svg-fallback ({used_model})"


def _extract_fenced_blocks(text: str) -> List[tuple[str, str]]:
    return [(lang.strip().lower(), body.strip()) for lang, body in re.findall(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```", text)]


def _export_chatbot_output(
    response_text: str,
    fmt: str,
    out_dir: Path,
    router: FreeLLMRouter,
) -> List[tuple[Path, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: List[tuple[Path, str]] = []

    if fmt == "chat":
        return saved

    if fmt == "md":
        p = out_dir / f"output_{stamp}.md"
        p.write_text(response_text, encoding="utf-8")
        saved.append((p, "markdown"))
        return saved

    if fmt == "pdf":
        pdf_path, engine = _render_markdown_to_pdf(response_text, out_dir / f"output_{stamp}.pdf")
        saved.append((pdf_path, engine))
        return saved

    if fmt == "docx":
        docx_path, engine = _render_markdown_to_docx(response_text, out_dir / f"output_{stamp}.docx")
        saved.append((docx_path, engine))
        return saved

    if fmt == "html":
        blocks = _extract_fenced_blocks(response_text)
        html_body = next((b for lang, b in blocks if lang in ("html", "htm")), None)
        if not html_body:
            html_body = _markdown_to_html(response_text, title=f"Output {stamp}")
        p = out_dir / f"output_{stamp}.html"
        p.write_text(html_body, encoding="utf-8")
        saved.append((p, "html"))
        return saved

    if fmt in ("json", "csv"):
        blocks = _extract_fenced_blocks(response_text)
        payload = next((b for lang, b in blocks if lang == fmt), response_text.strip())
        p = out_dir / f"output_{stamp}.{fmt}"
        p.write_text(payload, encoding="utf-8")
        saved.append((p, fmt))
        return saved

    if fmt == "code":
        blocks = _extract_fenced_blocks(response_text)
        if not blocks:
            p = out_dir / f"output_{stamp}.txt"
            p.write_text(response_text, encoding="utf-8")
            saved.append((p, "text"))
            return saved
        for idx, (lang, code_body) in enumerate(blocks, 1):
            ext = CODE_LANG_EXT.get(lang, ".txt")
            fname_match = re.search(r"^(?:#|//|<!--)\s*filename:\s*([^\s>]+)", code_body, re.IGNORECASE)
            fname = Path(fname_match.group(1)).name if fname_match else f"code_{stamp}_{idx}{ext}"
            p = out_dir / fname
            p.write_text(code_body + "\n", encoding="utf-8")
            saved.append((p, f"code:{lang or 'plain'}"))
        return saved

    if fmt == "videotext":
        md_path = out_dir / f"videotext_{stamp}.md"
        md_path.write_text(response_text, encoding="utf-8")
        saved.append((md_path, "videotext-script"))
        blocks = _extract_fenced_blocks(response_text)
        srt_body = next((b for lang, b in blocks if lang == "srt"), None)
        if srt_body:
            srt_path = out_dir / f"videotext_{stamp}.srt"
            srt_path.write_text(srt_body + "\n", encoding="utf-8")
            saved.append((srt_path, "srt-subtitles"))
        return saved

    if fmt == "image":
        img_prompt = response_text.strip().strip('"')
        img_path, engine = _generate_free_image(img_prompt, out_dir / f"image_{stamp}.png", router)
        saved.append((img_path, engine))
        return saved

    return saved


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
) -> None:
    print("\n============================================================")
    print("  FREE LLM ROUTER — UNIVERSAL MULTIMODAL CHATBOT MODE")
    print("============================================================")
    router = FreeLLMRouter(env_path=env_path, cache_path=cache_path, force_refresh=force_refresh)
    out_dir = Path(output_dir)

    # Step 1: Load Sources
    source_items = list(preset_sources) if preset_sources is not None else _prompt_chatbot_sources(router)
    source_payload = _load_sources_payload(source_items, router=router)
    if source_payload["file_count"]:
        print(f"[Sources Loaded] {source_payload['summary']} (~{source_payload['tokens']:,} tokens)")
    else:
        print("[Sources Loaded] None (pure chat mode)")

    # Step 2: Select from the 10-Choice Pipeline
    has_images = bool(source_payload.get("image_paths"))
    if preset_option:
        router.configure_non_interactive(preset_option)
    else:
        router.select_provider_interactive(vision_only=has_images, preset_sources=source_payload if source_payload["file_count"] else None)

    # Step 3: Interactive Chat & Any-Output Loop
    active_format = preset_format if preset_format in SUPPORTED_OUTPUT_FORMATS else "auto"
    history: List[Dict[str, str]] = []
    last_response = ""
    sources_sent_in_history = False

    if not one_shot_prompt:
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

        response_text, used_model = router.query(
            full_turn_prompt,
            system_prompt=sys_instruction,
            image_paths=attach_imgs if attach_imgs else None,
            history=history,
        )
        last_response = response_text
        history.append({"role": "user", "content": full_turn_prompt})
        history.append({"role": "assistant", "content": response_text})

        print(f"\n--- Assistant ({used_model}) [Output: {target_fmt.upper()}] ---")
        print(response_text)
        print("-" * 60)

        if target_fmt != "chat":
            saved_files = _export_chatbot_output(response_text, target_fmt, out_dir, router)
            for fpath, engine in saved_files:
                print(f"[Exported] {fpath} (via {engine})")

        if one_shot_prompt is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="broutegpt — Multi-Provider Free LLM Router, Multimodal Chatbot & Autonomous DP Coding Agent"
    )
    parser.add_argument("--sources", nargs="*", default=None, help="Source files, folders, or URLs to preload (chatbot mode)")
    parser.add_argument("--option", default=None, help="Preselect pipeline option 1-10 or 'provider:model_id'")
    parser.add_argument("--format", dest="out_format", default="auto", choices=SUPPORTED_OUTPUT_FORMATS, help="Target output format")
    parser.add_argument("--out-dir", default="chatbot_outputs", help="Directory where exported outputs are saved")
    parser.add_argument("--prompt", default=None, help="Optional one-shot prompt (skips interactive chat loop)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached free models")

    # Autonomous Zero-Question DP Builder & Forensics Flags
    parser.add_argument("--build", default=None, help="Run autonomous zero-question DP project builder for this specification")
    parser.add_argument("--inputs", nargs="*", default=None, help="Input files/folders to stage into sample_inputs/ for --build")
    parser.add_argument("--project-dir", default="./broutegpt_project", help="Target workspace directory for --build / forensics")
    parser.add_argument("--max-dps", type=int, default=6, help="Maximum number of Decision Points to plan (default: 6)")
    parser.add_argument("--max-steps", type=int, default=14, help="Maximum build/debug steps per DP (default: 14)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted autonomous DP build in --project-dir")
    parser.add_argument("--timeline", action="store_true", help="Print shadow-Git checkpoint & model timeline for --project-dir")
    parser.add_argument("--inspect-step", type=int, default=None, help="Inspect unified diff & model forensics for a checkpoint step ID")
    parser.add_argument("--rollback", type=int, default=None, help="Rollback workspace files (and context) to a checkpoint step ID")
    parser.add_argument("--no-context-rollback", action="store_true", help="With --rollback, restore files only without rewinding event context")
    args = parser.parse_args()

    if any([
        args.build is not None,
        args.resume,
        args.timeline,
        args.inspect_step is not None,
        args.rollback is not None,
    ]):
        from broutegpt_agent import run_agent_cli

        run_agent_cli(
            build_goal=args.build,
            inputs=args.inputs,
            project_dir=args.project_dir,
            max_dps=args.max_dps,
            max_steps=args.max_steps,
            resume=args.resume,
            show_timeline=args.timeline,
            inspect_step_id=args.inspect_step,
            rollback_step_id=args.rollback,
            restore_context=not args.no_context_rollback,
            preset_option=args.option,
            force_refresh=args.refresh,
        )
    else:
        run_chatbot_cli(
            force_refresh=args.refresh,
            preset_sources=args.sources,
            preset_option=args.option,
            preset_format=args.out_format,
            output_dir=args.out_dir,
            one_shot_prompt=args.prompt,
        )

