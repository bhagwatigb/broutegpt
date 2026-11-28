"""
Backward-compatible CLI entrypoint and re-export shim for `broutegpt_core`.
Preserves `python3 broutegpt.py ...` and `from broutegpt import FreeLLMRouter, ...`.
"""

import requests

from broutegpt_core.cli import _prompt_chatbot_sources, _read_piped_stdin, main, run_chatbot_cli, run_doctor
from broutegpt_core.config import (
    DEFAULT_CACHE_FILE,
    PROVIDER_RATE_INFO,
    _is_valid_key,
    _load_env_file,
    _redact_secrets,
    get_or_prompt_api_key,
    load_all_api_keys,
)
from broutegpt_core.exporters import (
    CODE_LANG_EXT,
    SUPPORTED_OUTPUT_FORMATS,
    _build_format_system_instruction,
    _detect_output_format,
    _export_chatbot_output,
    _extract_fenced_blocks,
    _generate_free_image,
    _markdown_to_html,
    _render_markdown_to_docx,
    _render_markdown_to_pdf,
)
from broutegpt_core.router import (
    FreeLLMRouter,
    _deep_reasoning_score,
    _encode_image,
    _estimate_model_tps,
    _heuristic_output_estimate,
    _is_deep_reasoning_model,
    _model_fits_budget,
    _sort_models,
    _sort_models_by_speed,
)
from broutegpt_core.sources import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    _estimate_text_tokens,
    _extract_file_source,
    _extract_zip_xml_text,
    _fetch_url_source,
    _load_sources_payload,
    _transcribe_media_file,
)


if __name__ == "__main__":
    main()

