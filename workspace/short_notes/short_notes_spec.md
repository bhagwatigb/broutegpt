Refactor and extend `sample_inputs/pipeline.py` into a fully cloud-API-driven, zero-data-loss Short-Notes PDF Pipeline, then run it on all 3 lecture PDFs staged in `sample_inputs/`.

## 1. Hard Constraints
- Zero Local AI Dependencies: Completely remove Ollama and all local model hooks (`_discover_ollama`, Option 1, `localhost:11434`, and local inference). Filter out `provider == "ollama"` from cache and all ranked pools. Use ONLY cloud free-tier APIs via `.env` (Gemini, Groq, OpenRouter, NVIDIA NIM, Hugging Face) with automatic fallback rerouting.
- No Video-Script Output: Do NOT generate NotebookLM video-script `_unicode.txt` dumps as the final deliverable.
- Deliverables: Build the upgraded `pipeline.py` in the project root and execute it on the 3 PDFs in `sample_inputs/` so each PDF produces:
  1. Raw slide-by-slide Markdown (`workspace/transcription/<stem>_raw.md`)
  2. Condensed zero-data-loss Short Notes Markdown (`workspace/short_notes/<stem>_short_notes.md`)
  3. Compiled, print-ready PDF (`workspace/pdfs/<stem>_short_notes.pdf`)

## 2. Pipeline Stages (`extract`, `transcribe`, `condense`, `compile-pdf`, `run-all`)
1. `extract`: Render each PDF page to PNG via PyMuPDF (`fitz`) and index assets per lecture.
2. `transcribe`: Cloud Vision OCR only (compress images <= 2.8 MB). Extract 100% of slide text, tables, LaTeX math/formulas, citations, numeric stats, and figure descriptions. Tag each slide with `is_progressive_duplicate` (true for cumulative reveal duplicates) and `keep_figure` (true ONLY for irreplaceable scientific figures like SEM/TEM, gels, Western/IP blots, Coulter/size-distribution curves, fluorescence images, or gelatinisation plots; false for stock photos/text-only slides).
3. `condense`: Cloud text API condensation in context-safe chunks + cross-chunk deduplication:
   - Rule A (Verbatim MCQ Lock): Whenever content can generate word-for-word MCQs (exact definitions, enzyme/gene/mutant-to-phenotype mappings, receptor/TF/promoter cis-element tables, chemical classes, numeric stats/percentages/timelines/pH values, reaction equations, classic experiment author/year citations, gene dosage comparisons), DO NOT paraphrase or shorten the concept—preserve exact wording, symbols, values, and structure verbatim.
   - Rule B (Zero-Loss Compression): Only where Rule A does not apply, collapse progressive-reveal duplicate slides, remove filler/redundant recaps, and tighten prose into readable, compact bullets with zero data loss.
   - Visuals & Mermaid: Embed `keep_figure=true` slide images cleanly. Add Mermaid flowcharts/mind maps where helpful for multi-step pathways or taxonomies; quote all node labels containing parentheses, colons, Greek letters, commas, or arrows (`A["AGPase (first committed step)"]`), and never put raw LaTeX inside Mermaid nodes.
4. `compile-pdf`: Compile Markdown -> UTF-8 HTML + KaTeX + SVG-rendered Mermaid + embedded local PNG figures -> PDF via headless Chromium/Playwright or WeasyPrint. Use Unicode-complete CSS fonts (`'Noto Sans', 'DejaVu Sans', Arial, sans-serif`) so Greek letters, arrows, `°`, and `£` never cause Unicode errors or tofu boxes. Never use `pdflatex` or Latin-1 libraries (`fpdf`).

## 3. Mandatory Verification & Audit Gates
1. Verify zero references to `ollama` or `localhost:11434` remain in active routing paths.
2. Run a cloud-API zero-data-loss diff audit comparing each `<stem>_raw.md` against `<stem>_short_notes.md` and auto-patch any missing gene, enzyme, mutant, cis-element, number, or citation before compiling.
3. Verify zero unrendered `mermaid` code fences, zero empty SVGs, zero broken image paths, zero `\ufffd` replacement characters, and non-empty compiled PDFs for all 3 input lectures.
EOF
