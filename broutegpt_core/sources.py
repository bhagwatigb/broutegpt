import html as html_lib
import mimetypes
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

from .config import _is_valid_key, _redact_secrets

if TYPE_CHECKING:
    from .router import FreeLLMRouter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    char_est = int(len(text) / 3.8)
    word_est = int(len(text.split()) * 1.35)
    return max(char_est, word_est, 1)


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

