import base64
import datetime
import html as html_lib
import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, List

import requests

from .config import _is_valid_key, _redact_secrets

if TYPE_CHECKING:
    from .router import FreeLLMRouter


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


def _generate_free_image(image_prompt: str, out_path: Path, router: "FreeLLMRouter") -> tuple[Path, str]:
    configured_img_models = list(router.role_config.get("image_generation") or [])
    for cand in configured_img_models:
        prov = cand.get("provider", "")
        mid = cand.get("id", "")
        if prov == "huggingface" and _is_valid_key(os.getenv("HF_API_KEY")):
            try:
                resp = requests.post(
                    f"[https://router.huggingface.co/hf-inference/models/](https://router.huggingface.co/hf-inference/models/){mid}",
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
                    "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)",
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
                    f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){mid}:generateContent?key={os.getenv('GEMINI_API_KEY')}",
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
        endpoints.append((f"[https://gen.pollinations.ai/image/](https://gen.pollinations.ai/image/){encoded}", {"Authorization": f"Bearer {pol_key}"}, "pollinations:api"))
    endpoints.append((f"[https://image.pollinations.ai/prompt/](https://image.pollinations.ai/prompt/){encoded}?nologo=true", {}, "pollinations:legacy"))

    for url, headers, label in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=90)
            if resp.ok and resp.headers.get("content-type", "").startswith("image/") and len(resp.content) > 1000:
                out_path.write_bytes(resp.content)
                return out_path, label
        except Exception:
            continue

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
    router: "FreeLLMRouter",
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

