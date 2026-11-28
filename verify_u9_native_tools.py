import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broutegpt import FreeLLMRouter
from broutegpt_agent.code_editor import normalize_action_batch, parse_agent_action, validate_action_json
from broutegpt_agent.router_adapter import CloudFirstRouter
from broutegpt_agent.tool_schemas import (
    AGENT_TOOL_SPECS,
    normalize_native_tool_calls,
    to_gemini_tools,
    to_openai_tools,
)


class FakeHTTPResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def main() -> None:
    # 1. Verify OpenAI and Gemini tool schema converters
    oa_tools = to_openai_tools(AGENT_TOOL_SPECS)
    assert len(oa_tools) >= 8 and oa_tools[0]["type"] == "function"
    gm_tools = to_gemini_tools(AGENT_TOOL_SPECS)
    assert "functionDeclarations" in gm_tools[0] and len(gm_tools[0]["functionDeclarations"]) >= 8

    # 2. Verify native tool-call normalization into valid action JSON
    batch_json = normalize_native_tool_calls(
        [
            {"name": "write_file", "args": {"path": "a.py", "content": "x = 1\n"}},
            {"name": "run_shell", "args": {"command": "python3 a.py"}},
        ],
        thought_text="Writing and testing a.py",
    )
    assert validate_action_json(batch_json) is None
    parsed_batch = parse_agent_action(batch_json)
    assert len(normalize_action_batch(parsed_batch)) == 2

    # 3. Verify FreeLLMRouter._call_model sends Gemini functionDeclarations & OpenAI tools
    router = FreeLLMRouter.__new__(FreeLLMRouter)
    router.timeout = 30
    router.last_call_usage = {}

    captured_payloads = []

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_payloads.append((url, json))
        if "generativelanguage.googleapis.com" in url:
            return FakeHTTPResp({
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "Reading file via Gemini tool."},
                            {"functionCall": {"name": "read_file", "args": {"path": "main.py", "start_line": 1}}},
                        ]
                    },
                    "finishReason": "STOP",
                }]
            })
        return FakeHTTPResp({
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_dir",
                                "arguments": '{"path": "."}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }]
        })

    with patch("broutegpt.requests.post", side_effect=fake_post):
        gm_out = router._call_model("gemini", "gemini-2.5-flash", "Inspect", None, None, 0.1, 256, native_tools=AGENT_TOOL_SPECS)
        assert validate_action_json(gm_out) is None
        assert parse_agent_action(gm_out)["action"] == "read_file"
        assert "tools" in captured_payloads[-1][1]

        groq_out = router._call_model("groq", "llama-3.3-70b-versatile", "List", None, None, 0.1, 256, native_tools=AGENT_TOOL_SPECS)
        assert validate_action_json(groq_out) is None
        assert parse_agent_action(groq_out)["action"] == "list_dir"
        assert "tools" in captured_payloads[-1][1]

    # 4. Verify CloudFirstRouter fallback to JSON mode when cloud model rejects native tools & for Ollama
    mock_base = MagicMock()
    mock_base.models_by_option = {
        "1": [{"id": "qwen2.5-coder:7b", "provider": "ollama", "input_limit": 8192, "output_limit": 4096, "chat": True}],
        "2": [{"id": "gemma-3-27b-it", "provider": "gemini", "input_limit": 32768, "output_limit": 4096, "chat": True}],
    }
    mock_base.role_config = {}
    mock_base._cooldown_until = {}
    mock_base.last_call_usage = {}

    call_kwargs = []

    def fake_router_call(prov, mid, prompt, *a, **kw):
        call_kwargs.append((prov, mid, kw.get("native_tools")))
        return '{"thought":"fallback json","action":"list_dir","path":"."}'

    mock_base._call_model.side_effect = fake_router_call
    cfr = CloudFirstRouter(router=mock_base)
    res_txt, used = cfr.query_autonomous("List workspace", tier="speed", use_native_tools=True)
    assert used == "gemini:gemma-3-27b-it"
    assert call_kwargs[-1][2] is None, "Gemma/Ollama non-tool models must automatically fall back to JSON mode"
    assert parse_agent_action(res_txt)["action"] == "list_dir"

    print("VERIFY_U9_NATIVE_TOOLS: PASS")


if __name__ == "__main__":
    main()

