# `broutegpt` (v0.2.0)

**Zero-Cost Multi-Provider Free LLM Router, Universal Multimodal Chatbot & Autonomous Decision-Point (DP) Software Engineering Agent**

`broutegpt` unifies local Ollama models and free-tier cloud LLM providers (Google Gemini, Groq, OpenRouter, NVIDIA NIM, and Hugging Face) behind a single intelligent routing engine, a multimodal document/media chatbot with multi-format artifact export, and a zero-question autonomous coding agent powered by parallel `git worktree` DAG execution, shadow-Git time-travel checkpoints, OS-level sandboxing, deterministic lifecycle hooks, and Model Context Protocol (MCP) support.

---

## Table of Contents

1. [Core Capabilities & Features](#core-capabilities--features)
2. [Architecture & Repository Layout](#architecture--repository-layout)
3. [Installation & Setup](#installation--setup)
4. [User Manual](#user-manual)
   - [1. Environment Diagnostics (`--doctor`) & Provider Setup](#1-environment-diagnostics---doctor--provider-setup)
   - [2. The 10-Choice Free LLM Routing Pipeline](#2-the-10-choice-free-llm-routing-pipeline)
   - [3. Interactive Multimodal Chatbot & Multi-Format Exporter](#3-interactive-multimodal-chatbot--multi-format-exporter)
   - [4. Unix Piping, Headless Scripting & CI Modes](#4-unix-piping-headless-scripting--ci-modes)
   - [5. Autonomous Zero-Question Decision-Point (DP) Builder](#5-autonomous-zero-question-decision-point-dp-builder)
   - [6. Parallel DAG Execution via `git worktree` Isolation](#6-parallel-dag-execution-via-git-worktree-isolation)
   - [7. Shadow-Git Time-Travel, Diff Forensics, Rewind & Branching](#7-shadow-git-time-travel-diff-forensics-rewind--branching)
   - [8. Two-Tier Project Memory (`AGENTS.md` + `MEMORY.md`)](#8-two-tier-project-memory-agentsmd--memorymd)
   - [9. OS Sandbox Hardening, Credential Scrubbing & Permission Policies](#9-os-sandbox-hardening-credential-scrubbing--permission-policies)
   - [10. Deterministic Lifecycle Hooks (`hooks.json`)](#10-deterministic-lifecycle-hooks-hooksjson)
   - [11. Model Context Protocol (MCP) Client & `--mcp` Server Mode](#11-model-context-protocol-mcp-client--mcp-server-mode)
5. [Complete CLI Reference](#complete-cli-reference)
6. [Verification & Regression Suite](#verification--regression-suite)

---

## Core Capabilities & Features

### Multi-Provider Free LLM Router (`broutegpt_core`)
- **6 Unified Backends:** Routes seamlessly across **Local Ollama**, **Google Gemini (Free Tier)**, **Groq**, **OpenRouter (`:free` models)**, **NVIDIA NIM**, and **Hugging Face Inference Router**.
- **Automated Model Discovery & Reconfiguration Drive:** Discovers live free models, filters out deprecated/preview endpoints, benchmarks throughput (`tok/s`), and automatically assigns specialized models for `image_transcription`, `image_generation`, `audio_transcription`, `fast_triage`, and `deep_synthesis`.
- **Smart Task-Sized Optimization (Option 10):** Measures combined prompt + source + image token volume, estimates output token requirements via a fast triage model, eliminates models that would suffer input/output context overflow, and ranks eligible models by speed or deep reasoning capability.
- **Resilient Automatic Failover:** Detects `429` rate limits (applying a 60-second per-model cooldown), mid-code-block `MAX_TOKENS` truncations, degenerate repetition loops, and schema validation failures—rerouting transparently to the next ranked model.

### Universal Multimodal Chatbot & Artifact Exporter
- **Any-Input Ingestion:** Reads PDFs (with automatic PyMuPDF page rendering and vision OCR for scanned PDFs), DOCX, PPTX, XLSX, Markdown, plain text, source code, live web URLs, images (`PNG`, `JPG`, `WEBP`, `GIF`, `BMP`), audio/video files (`MP3`, `WAV`, `M4A`, `MP4`, `MOV` via Whisper transcription), entire directories, and inline `@path` mentions.
- **Multi-Format Artifact Export:** Exports responses directly into `.pdf`, `.docx`, `.html`, `.md`, runnable multi-file source code (`# filename: ...`), `.json`, `.csv`, `videotext` packages (scene-by-scene narration script + `.srt` subtitles), or generated images (`.png` / `.svg`).

### Autonomous Decision-Point (DP) Coding Agent (`broutegpt_agent`)
- **Zero-Question Architectural Planning:** Decomposes specifications into 2–6 self-contained Decision Points (`.broutegpt/dp_plan.json`), each gated by an executable Python verification script (`verify_dp1.py`, `verify_dp2.py`, ...).
- **Parallel DAG Execution:** Executes independent Decision Points concurrently in isolated `.broutegpt/worktrees/dp-<id>` Git worktrees, verifies each branch independently, performs 3-way merges, and falls back gracefully to sequential execution if branches conflict on overlapping lines.
- **Asymmetric Model Escalation & On-Demand Advisor:** Uses deep reasoning models for initial synthesis and heavy debugging, fast models for verification and surgical SEARCH/REPLACE patches, and provides an on-demand `consult_advisor` tool plus an isolated read-only `ExploreSubagent` (`spawn_subagent`) to prevent context pollution.
- **Hybrid Native Tool-Calling:** Uses native OpenAI `tools` and Gemini `functionDeclarations` on supported cloud providers while falling back cleanly to structured JSON actions on local Ollama models.
- **Read-Before-Edit Guards & Static Validation:** Tracks file SHA-256 hashes to prevent stale overwrites, rolls back syntax-invalid writes/patches automatically, and validates Python, JSON, YAML, TOML, and TypeScript/TSX syntax before accepting edits.

### Security, Governance & Extensibility
- **Hardened OS Sandbox:** Jails shell commands inside macOS `sandbox-exec` (Seatbelt) or Linux `bwrap` (Bubblewrap), protecting `.git/hooks`, `.git/config`, and `.broutegpt/` from tampering, blocking reads of `~/.ssh`, `~/.aws`, and `~/.gnupg`, and scrubbing API keys/tokens from child process environments.
- **Declarative Permission Rules & `--auto-guard`:** Evaluates `.broutegpt/settings.json` `allow` / `ask` / `deny` glob policies across chained shell operators (`&&`, `||`, `;`, `|`), with optional free-model safety classification (`--auto-guard`).
- **Deterministic Lifecycle Hooks:** Executes custom shell hooks on `SessionStart`, `PreToolUse`, `PostToolUse`, `PostToolBatch`, `PreCompact`, `PreModelSwitch`, `DPVerifyFailed`, and `Stop`.
- **MCP Client & Server:** Connects to external stdio MCP servers with lazy schema hydration (`load_mcp_tool` -> `call_mcp_tool`) to save tokens, and runs as a stdio MCP server via `broutegpt --mcp`.

---

## Architecture & Repository Layout

```text
broutegpt/
├── broutegpt.py                      # Backward-compatible CLI entrypoint & re-export shim
├── pyproject.toml                    # PEP 621 package metadata & `broutegpt` console script
├── .env                              # Local API keys (never committed or overwritten on sync)
├── .free_models_cache.json           # Cached model catalog, speed benchmarks & role assignments
│
├── broutegpt_core/                   # Core Router, Multimodal Ingestion, Exporters & CLI
│   ├── __init__.py
│   ├── config.py                     # .env loader, API key validation & secret redaction
│   ├── sources.py                    # Multimodal file/folder/URL/image/audio/PDF extractor
│   ├── router.py                     # FreeLLMRouter (10-option pipeline, fallback & sizing)
│   ├── exporters.py                  # PDF, DOCX, HTML, Code, JSON, CSV, VideoText & Image export
│   └── cli.py                        # Interactive chatbot, Unix pipe, --plan, --doctor & main()
│
├── broutegpt_agent/                  # Autonomous DP Coding Agent & Governance Engine
│   ├── __init__.py
│   ├── dp_orchestrator.py            # DAG wave scheduler, worktree runner & DP build/debug loop
│   ├── checkpoint_manager.py         # Shadow-Git snapshots, worktrees, diff forensics & branching
│   ├── code_editor.py                # SHA-256 read guards, SEARCH/REPLACE patching & static linter
│   ├── condenser.py                  # Dynamic token-budgeted rolling context condenser & MEMORY.md
│   ├── event_stream.py               # Append-only structured event log with rewind support
│   ├── hook_engine.py                # Deterministic JSON lifecycle hook runner (hooks.json)
│   ├── input_inspector.py            # Stage 0 external input staging & schema inspector
│   ├── mcp_bridge.py                 # Lazy stdio MCP client registry & `--mcp` JSON-RPC server
│   ├── permissions.py                # Glob permission policy & --auto-guard safety classifier
│   ├── repo_map.py                   # AST/symbol repository mapper, glob_files & grep_search
│   ├── router_adapter.py             # Tiered autonomous routing & token telemetry wrapper
│   ├── subagent.py                   # Isolated read-only ExploreSubagent
│   ├── terminal_executor.py          # Seatbelt/bwrap OS sandbox & credential scrubber
│   ├── tool_schemas.py               # Native OpenAI & Gemini function-calling schemas
│   └── cli_runner.py                 # Autonomous builder & forensics CLI dispatcher
│
└── verify_u1_*.py .. verify_u15_*.py # Cumulative 15-stage verification & regression test suite
```

---

## Installation & Setup

### 1. Clone & Create a Virtual Environment

```bash
git clone https://github.com/bhagwatigb/broutegpt.git
cd broutegpt
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install the core package (editable mode registers the `broutegpt` CLI command):

```bash
pip install -e .
```

To enable native PDF, DOCX, PPTX, XLSX, and Markdown rendering extras:

```bash
pip install -e ".[docs]"
```

### 3. Configure API Keys (`.env`)

Create a `.env` file in the repository root with any combination of free-tier API keys (and/or run Ollama locally on `http://localhost:11434`):

```env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
NVIDIA_API_KEY="your_nvidia_api_key"
HF_API_KEY="your_huggingface_api_key"
OLLAMA_BASE_URL="http://localhost:11434"
```

---

## User Manual

### 1. Environment Diagnostics (`--doctor`) & Provider Setup

Run `--doctor` at any time to verify Python, Git, OS sandboxing (`macos-seatbelt` or `linux-bwrap`), Ollama reachability, masked API keys, and cached model readiness:

```bash
python3 broutegpt.py --doctor
# Or emit machine-readable JSON:
python3 broutegpt.py --doctor --json
```

To force a fresh discovery of all available free models across configured providers and re-run the automated role assignment drive:

```bash
python3 broutegpt.py --refresh
```

---

### 2. The 10-Choice Free LLM Routing Pipeline

When launched interactively (or preselected via `--option <1-10>` or `--option provider:model_id`), `broutegpt` offers 10 routing strategies:

| Option | Pipeline Mode | Behavior |
| :--- | :--- | :--- |
| **`1`** | **Local Ollama Models** | Routes strictly to locally running Ollama models (`100%` offline/private). |
| **`2`** | **Google Gemini (Free Tier)** | Routes across stable Gemini Flash / Gemma models sorted by context capacity. |
| **`3`** | **Groq (Free Tier)** | Routes across high-speed Groq LPU models (including Whisper audio models). |
| **`4`** | **OpenRouter (Free Models)** | Routes across zero-cost `:free` OpenRouter models. |
| **`5`** | **Other Free APIs** | Routes across NVIDIA NIM and Hugging Face free inference endpoints. |
| **`6`** | **Compare by Token Limits** | Displays a side-by-side table of the Top 10 models ranked by `input_limit + output_limit`. |
| **`7`** | **Top-10 Token-Ranked Fallback** | Automatically tries Rank 1 -> Rank 10 largest-context models across all providers. |
| **`8`** | **Top-10 Speed-Ranked Fallback** | Ranks models by output speed (`tok/s`), with option `b` to run live latency benchmarks. |
| **`9`** | **Custom Model** | Targets a specific `provider:model_id` (e.g., `groq:llama-3.3-70b-versatile`) with menu fallback on error. |
| **`10`** | **Smart Task-Sized Optimizer** | Computes exact prompt + source tokens, sizes output budget, filters out models that would overflow, and picks the fastest or deepest sufficient model. |

Additional interactive menu shortcuts:
- **`r`**: Refresh live model catalogs across all providers and run the Reconfiguration Drive.
- **`c`**: View or re-run the Reconfiguration Drive on the existing `.free_models_cache.json`.

---

### 3. Interactive Multimodal Chatbot & Multi-Format Exporter

Launch interactive chatbot mode with optional preloaded files, folders, or URLs:

```bash
python3 broutegpt.py --sources ./docs/spec.pdf https://example.com/api-notes
```

#### Interactive Slash Commands
Inside the chat loop, use slash commands to manage sources, switch pipelines, export artifacts, or launch an autonomous code build:

- `/sources` — Add more files/folders/URLs (`a`), clear all attached sources (`c`), or inspect token counts.
- `/pipeline` — Switch between routing options `1`–`10` mid-conversation.
- `/out <fmt>` — Set active output format (`auto`, `chat`, `md`, `pdf`, `code`, `image`, `videotext`, `html`, `docx`, `json`, `csv`).
- `/save <fmt>` — Export the previous assistant response immediately to `./chatbot_outputs/` in the requested format.
- `/build <goal>` — Hand off current conversation context and attached sources directly to the Autonomous DP Builder.
- `/clear` — Clear conversation history.
- `/exit` — Exit the session.

#### Inline `@path` Attachment & Auto-Format Detection
You can reference files inline in any prompt (e.g., `Summarize @./notes/lecture.pdf and export as pdf`)—`broutegpt` automatically attaches the file, detects the target format (`pdf`), renders the document into `./chatbot_outputs/`, and prints the saved file path.

---

### 4. Unix Piping, Headless Scripting & CI Modes

`broutegpt` integrates cleanly into shell pipelines and CI workflows:

```bash
# Pipe stdin directly into a headless prompt
git diff | python3 broutegpt.py -p "Write a concise conventional commit message" --print

# Emit structured JSON (all routing logs redirect to stderr; stdout is pure JSON)
cat traceback.log | python3 broutegpt.py -p "Identify root cause and fix" --json

# Read-only architectural planning without modifying files
python3 broutegpt.py --plan --build "Build a FastAPI webhook relay with SQLite queue" --project-dir ./relay_app --json

# Bare CI fast-start mode (skips interactive menus, AGENTS.md, and MEMORY.md auto-loading)
python3 broutegpt.py --bare -p "Return OK if valid JSON: {\"a\": 1}" --print
```

---

### 5. Autonomous Zero-Question Decision-Point (DP) Builder

To build and verify a complete multi-file software project autonomously from start to finish:

```bash
python3 broutegpt.py \
  --build "Build a CLI log analyzer supporting JSON/CSV output, regex filtering, and summary stats" \
  --inputs ./sample_logs/ \
  --project-dir ./log_analyzer_ws \
  --max-dps 4 \
  --max-steps 14
```

#### How the Autonomous Loop Works
1. **Stage 0 (Staging & Planning):** Copies `--inputs` into `<project-dir>/sample_inputs/`, inspects schemas, builds a symbol `RepoMap`, and generates `.broutegpt/dp_plan.json`.
2. **Build & Batch Execution:** In each DP, the agent executes single actions or multi-action batches (up to 6 sequential actions per turn: `write_file`, `patch_file`, `read_file`, `grep_search`, `glob_files`, `consult_advisor`, `spawn_subagent`, `run_shell`, `load_mcp_tool`, `call_mcp_tool`).
3. **Static Linter & Verification Gate:** Every write/patch is statically syntax-checked (rolling back broken writes immediately). Before `dp_complete` is accepted, `verify_dp<k>.py` **and** all prior `verify_dp1.py .. verify_dp<k-1>.py` scripts must exit `0`.
4. **Escalation & Auto-Rollback:** Fast models handle routine edits and test runs; if 3 consecutive failures occur, routing escalates to `deep` reasoning models. At 4 consecutive failures, workspace files automatically roll back to the clean DP start checkpoint while retaining error diagnostics in context.
5. **Resume Interrupted Builds:**
   ```bash
   python3 broutegpt.py --resume --project-dir ./log_analyzer_ws
   ```

---

### 6. Parallel DAG Execution via `git worktree` Isolation

`dp_plan.json` supports an optional `depends_on` list of prerequisite DP IDs on each Decision Point:

```json
{
  "project_name": "analytics_suite",
  "entrypoint": "cli.py",
  "dps": [
    { "id": 1, "title": "CSV & JSON Ingestion Engine", "depends_on": [] },
    { "id": 2, "title": "Statistical Aggregation Core", "depends_on": [] },
    { "id": 3, "title": "Unified CLI & Report Exporter", "depends_on": [1, 2] }
  ]
}
```

- **Wave 1 (`[1, 2]`):** `DPOrchestrator` provisions isolated Git worktrees at `.broutegpt/worktrees/dp-1` and `.broutegpt/worktrees/dp-2`, executes and verifies both DPs concurrently on separate threads, performs 3-way merges back into the workspace, runs the post-merge regression suite, and tags `dp-1-verified` and `dp-2-verified`.
- **Conflict Fallback:** If two parallel branches modify overlapping lines in the same file, the conflicting merge aborts cleanly without conflict markers and re-runs that DP sequentially on top of the merged workspace.
- **Wave 2 (`[3]`):** Runs once both prerequisite DPs are merged and verified.

---

### 7. Shadow-Git Time-Travel, Diff Forensics, Rewind & Branching

Every file write, patch, and shell command is snapshotted in a hidden shadow Git repository (`.broutegpt/shadow_git`) so your project's main `.git` history stays clean and only receives verified DP milestone commits and tags (`dp-<k>-verified`).

```bash
# View chronological timeline of all steps, models used, and verified milestones
python3 broutegpt.py --timeline --project-dir ./log_analyzer_ws

# Inspect unified diff, event payload, and model attribution for Step #7
python3 broutegpt.py --inspect-step 7 --project-dir ./log_analyzer_ws

# Rewind both files and conversation events to Step #5
python3 broutegpt.py --rollback 5 --scope both --project-dir ./log_analyzer_ws

# Rewind ONLY code files (keep diagnostic chat history) or ONLY chat events (keep files)
python3 broutegpt.py --rollback 5 --scope code --project-dir ./log_analyzer_ws
python3 broutegpt.py --rollback 5 --scope chat --project-dir ./log_analyzer_ws

# Fork an isolated experimental workspace branch at Step #5 without mutating the main workspace
python3 broutegpt.py --branch 5 --branch-name alt-parser --project-dir ./log_analyzer_ws
```

---

### 8. Two-Tier Project Memory (`AGENTS.md` + `MEMORY.md`)

Unless `--bare` is passed, `DPOrchestrator` automatically loads hierarchical project instructions and cross-session memory:
1. **Repository & User Rule Files:** Loads `AGENTS.md`, `BROUTEGPT.md`, or `CLAUDE.md` from the workspace root and `~/.broutegpt/`. Supports cycle-safe recursive `@relative/path.md` file imports (up to depth 5).
2. **Bounded Cross-Session Memory (`.broutegpt/MEMORY.md`):** Automatically records verified DP summaries and architectural notes across sessions, capped at 200 lines / 25 KB.

---

### 9. OS Sandbox Hardening, Credential Scrubbing & Permission Policies

#### OS-Level Sandboxing
All `run_shell` commands execute inside a workspace-jailed sandbox (`sandbox-exec` on macOS, `bwrap` on Linux):
- Denies writes outside the active workspace and blocks writes to `.git/hooks`, `.git/config`, and `.broutegpt/`.
- Blocks read access to sensitive credential directories (`~/.ssh`, `~/.aws`, `~/.gnupg`).
- Strips environment variables matching `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `AWS_*`, `GITHUB_TOKEN`, etc. from child processes.
- Pass `--strict-sandbox` to disable network access inside the sandbox (`(deny network*)` / `--unshare-net`).
- Set `BROUTEGPT_NO_SANDBOX=1` only if running inside an already-isolated container that lacks nested namespace privileges.

#### Declarative Permission Policies (`.broutegpt/settings.json`)
Configure glob rules evaluated across shell command chains (`&&`, `||`, `;`, `|`):

```json
{
  "permissions": {
    "allow": [
      "run_shell:python3 *",
      "run_shell:pytest *"
    ],
    "ask": [
      "run_shell:pip install *"
    ],
    "deny": [
      "run_shell:rm -rf *",
      "run_shell:curl * | bash"
    ]
  }
}
```

Add `--auto-guard` on the CLI to screen unapproved shell commands with a fast free-tier safety classifier before execution.

---

### 10. Deterministic Lifecycle Hooks (`hooks.json`)

Define deterministic shell hooks in `.broutegpt/hooks.json` (or `~/.broutegpt/hooks.json`) to enforce invariants or inject dynamic context:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "run_shell",
        "command": "python3 scripts/check_policy.py"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "write_file",
        "command": "echo 'Remember to keep functions typed.'"
      }
    ]
  }
}
```

- **Supported Events:** `SessionStart`, `PreToolUse`, `PostToolUse`, `PostToolBatch`, `PreCompact`, `PreModelSwitch`, `DPVerifyFailed`, `Stop`.
- **Exit Code `0`:** Succeeds; any `stdout` is injected into the agent's observation stream as `additionalContext`.
- **Exit Code `2`:** Deterministically blocks the tool call or session start and feeds `stderr` back to the agent as the block reason.

---

### 11. Model Context Protocol (MCP) Client & `--mcp` Server Mode

#### Connecting External MCP Servers (Lazy Tool Search)
Register stdio MCP servers in `.broutegpt/settings.json`:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./data.db"]
    }
  }
}
```

To prevent large MCP tool catalogs from consuming context tokens, `broutegpt` injects only compact tool names and one-line descriptions into the prompt. The agent calls `load_mcp_tool` to hydrate the full JSON `inputSchema` on demand before invoking `call_mcp_tool`.

#### Running `broutegpt` as an MCP Server (`--mcp`)
Expose `broutegpt` tools (`broutegpt_doctor`, `broutegpt_repo_map`, `broutegpt_plan`, `broutegpt_chat`) over stdio JSON-RPC 2.0 to Claude Desktop, Cursor, or any MCP-compatible host:

```json
{
  "mcpServers": {
    "broutegpt": {
      "command": "python3",
      "args": ["/path/to/broutegpt/broutegpt.py", "--mcp", "--project-dir", "/path/to/workspace"]
    }
  }
}
```

---

## Complete CLI Reference

| Flag | Description |
| :--- | :--- |
| `-p`, `--prompt <text>` | Execute a single one-shot prompt and exit. |
| `--print` | Headless print mode: suppresses banners/logs on `stdout` and prints only the clean response. |
| `--output-format <text\|json>` | Format `stdout` as human-readable `text` (default) or structured `json`. |
| `--json` | Shorthand for `--output-format json`. |
| `--bare` | CI fast-start mode: skips interactive menus and `AGENTS.md` / `MEMORY.md` rule loading. |
| `--plan` | Stage 0 read-only mode: generates `.broutegpt/dp_plan.json` without modifying source files. |
| `--doctor` | Run environment, Git, OS sandbox, API key, Ollama, and model cache diagnostics. |
| `--mcp` | Run `broutegpt` as a stdio JSON-RPC 2.0 MCP server. |
| `--sources <paths/urls...>` | Preload files, folders, or URLs in chatbot mode. |
| `--option <1-10 \| prov:id>` | Preselect a routing pipeline option or custom `provider:model_id`. |
| `--format <fmt>` | Export artifact format (`auto`, `chat`, `md`, `pdf`, `code`, `image`, `videotext`, `html`, `docx`, `json`, `csv`). |
| `--out-dir <dir>` | Output directory for exported chatbot artifacts (default: `./chatbot_outputs`). |
| `--refresh` | Force refresh the cached free model catalog and re-run the Reconfiguration Drive. |
| `--build <spec>` | Run the autonomous zero-question DP project builder for `<spec>`. |
| `--inputs <paths...>` | Stage external input files/folders into `<project-dir>/sample_inputs/` for `--build`. |
| `--project-dir <dir>` | Target workspace directory for `--build`, `--plan`, `--mcp`, and forensics (default: `./broutegpt_project`). |
| `--max-dps <int>` | Maximum number of Decision Points to plan (default: `6`). |
| `--max-steps <int>` | Maximum build/debug steps per Decision Point (default: `14`). |
| `--resume` | Resume an interrupted autonomous build from `.broutegpt/dp_plan.json`. |
| `--timeline` | Print the shadow-Git checkpoint & model attribution timeline. |
| `--inspect-step <id>` | Show unified diff, event details, and model forensics for checkpoint `<id>`. |
| `--rollback <id>` | Restore workspace files and/or event context to checkpoint `<id>`. |
| `--scope <both\|code\|chat>` | Rollback scope: `both` (default), `code` (files only), or `chat` (events only). |
| `--branch <id>` | Fork an isolated session directory under `.broutegpt/branches/` from checkpoint `<id>`. |
| `--branch-name <name>` | Custom branch name used with `--branch`. |
| `--strict-sandbox` | Block network access inside sandboxed shell commands. |
| `--auto-guard` | Screen unapproved shell commands using a fast free-model safety classifier. |

---

## Verification & Regression Suite

The repository includes an end-to-end verification suite (`verify_u2_*.py` through `verify_u15_*.py`) testing every subsystem offline with deterministic mocks as well as live smoke validation:

```bash
# Run the full cumulative release & regression suite
python3 verify_u15_release.py

# Or run individual subsystem gates directly:
python3 verify_u4_sandbox.py        # Seatbelt/bwrap sandbox & credential scrubbing
python3 verify_u7_hooks.py          # Deterministic lifecycle hook engine
python3 verify_u8_permissions.py    # Declarative permission globs & --auto-guard
python3 verify_u12_worktrees.py     # Parallel DAG execution & git worktree 3-way merge
python3 verify_u13_cli_ux.py        # Unix piping, --print, --json, --bare, --plan & --doctor
python3 verify_u14_mcp.py           # Lazy MCP client & --mcp JSON-RPC server
```