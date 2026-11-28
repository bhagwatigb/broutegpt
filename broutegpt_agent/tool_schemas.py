import json
from typing import Any, Dict, List


AGENT_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "write_file",
        "description": "Write a complete file inside the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Concise reasoning for this action."},
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Complete UTF-8 file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "patch_file",
        "description": "Apply surgical SEARCH/REPLACE edits to an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Concise reasoning for this patch."},
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search": {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["search", "replace"],
                    },
                },
                "fallback_content": {"type": "string"},
            },
            "required": ["path", "edits"],
        },
    },
    {
        "name": "read_file",
        "description": "Read numbered lines of a workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories inside a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "path": {"type": "string"},
            },
        },
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command inside the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search workspace files with a regex pattern and context lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "max_matches": {"type": "integer"},
                "context_lines": {"type": "integer"},
                "glob": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob_files",
        "description": "List workspace files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "pattern": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "consult_advisor",
        "description": "Consult a deep-reasoning advisor model on demand for architectural guidance or tricky debugging without escalating the whole DP.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "question": {"type": "string", "description": "Specific technical question or traceback diagnosis request for the deep advisor."},
                "context": {"type": "string", "description": "Optional relevant code snippet or error excerpt."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "spawn_subagent",
        "description": "Spawn an isolated read-only explore subagent to inspect files, search symbols, and return only a concise summary without bloating main context.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "task": {"type": "string", "description": "What codebase question, module structure, or symbol usages to explore."},
                "path": {"type": "string", "description": "Optional workspace-relative subdirectory to focus exploration."},
                "max_turns": {"type": "integer", "description": "Maximum read-only exploration turns (default 6)."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "dp_complete",
        "description": "Mark the current Decision Point complete after verify_dp<k>.py exists and passes.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
]


def to_openai_tools(specs: List[Dict[str, Any]] = AGENT_TOOL_SPECS) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for s in specs
    ]


def to_gemini_tools(specs: List[Dict[str, Any]] = AGENT_TOOL_SPECS) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    for s in specs:
        declarations.append({
            "name": s["name"],
            "description": s.get("description", ""),
            "parameters": s.get("parameters", {"type": "object", "properties": {}}),
        })
    return [{"functionDeclarations": declarations}]


def normalize_native_tool_calls(
    calls: List[Dict[str, Any]],
    thought_text: str = "",
) -> str:
    """
    Convert normalized [{name, args}] tool calls into the canonical JSON action
    envelope accepted by parse_agent_action / normalize_action_batch.
    """
    actions: List[Dict[str, Any]] = []
    for item in calls:
        name = str(item.get("name") or "").strip().lower()
        raw_args = item.get("args") or {}
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        entry: Dict[str, Any] = {"action": name, **raw_args}
        actions.append(entry)

    lead_thought = thought_text.strip() or (
        str(actions[0].get("thought") or "").strip() if actions else ""
    ) or f"Executing {actions[0]['action']} via native tool call."
    if len(actions) == 1:
        single = dict(actions[0])
        single.setdefault("thought", lead_thought)
        return json.dumps(single, ensure_ascii=False)
    return json.dumps({"thought": lead_thought, "actions": actions}, ensure_ascii=False)

