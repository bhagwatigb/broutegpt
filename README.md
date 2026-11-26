# broutegpt

**Multi-Provider Free LLM Router, Multimodal Chatbot, and Autonomous Zero-Question Decision-Point (DP) Coding Agent.**

`broutegpt` combines a smart multi-provider free-tier LLM router (Gemini, Groq, OpenRouter, NVIDIA NIM, Hugging Face, and local Ollama) with an autonomous software engineering agent capable of planning, building, verifying, debugging, and Git-checkpointing complete projects from a single prompt and optional input files—with zero human intervention.

---

## Key Features

### 1. Autonomous Zero-Question Decision-Point (DP) Builder (`--build`)
* **Stage 0 Input Staging & Schema Inspection:** Automatically copies user-supplied input files or directories (`--inputs`) into `<project_dir>/sample_inputs/` and extracts structural schemas (CSV/TSV headers, JSON keys, FASTA/text previews) for the architect model.
* **Master Execution Blueprint (`dp_plan.json`):** Decomposes complex specifications into 2–6 sequential, self-contained Decision Points, each paired with concrete deliverables and a standalone verification script (`verify_dp1.py`, `verify_dp2.py`, ...).
* **Build → Lint → Verify → Debug → Regression Guard Loop:**
  * Every file write or surgical `SEARCH/REPLACE` patch passes an instant static/AST syntax pre-check before touching disk.
  * `dp_complete` is mechanically gated: the orchestrator executes `verify_dp1.py` through `verify_dp<k>.py` itself and rejects premature completion if any assertion or prior DP regresses.
  * Failures automatically feed tracebacks back into the agent and escalate model tiers (`smart` → `deep` → `max_context`), auto-rolling back workspace files if a debug branch thrashes.
* **Clean Live Console Narration:** Transient routing and fallback attempts overwrite a single terminal status line and auto-erase on success, printing only the winning model, latency, and step-by-step findings (`Finding/Plan`, `Created/Wrote`, `Debug/Patched`, `Executed`, `Problem Found`).

### 2. Cloud-First Tiered Routing with Last-Resort Local Ollama
* Exhausts active, coding-capable cloud models across your configured free API keys (**Gemini**, **Groq**, **OpenRouter**, **NVIDIA NIM**, **Hugging Face**) with automatic per-model and per-provider cooldown tracking (`429`, `404`, auth errors, malformed JSON).
* Local **Ollama** is isolated strictly as a last-resort fallback (auto-starting `ollama serve` in the background on macOS or bridging via `host.docker.internal:11434` inside Docker) only if every eligible cloud model is unavailable.

### 3. Shadow-Git Time-Travel Forensics
* **Dual-Git Architecture:**
  * A hidden shadow Git repository (`.broutegpt/shadow_git`) snapshots workspace state on every file write, patch, and shell command without polluting your main commit log.
  * Your project's main Git repository receives clean milestone commits and annotated tags (`dp-1-verified`, `dp-2-verified`, ...) only when a DP passes verification.
* **Time-Travel Commands:** Inspect the step-by-step timeline (`--timeline`), view unified diffs and model attribution (`--inspect-step`), or rewind workspace files and agent memory to any prior step (`--rollback` + `--resume`).

### 4. Rolling Context Condensation & AST Repo Map
* **Persistent Event Stream (`events.jsonl`) + `RollingCondenser`:** Dynamically summarizes older steps when token budgets tighten, allowing seamless mid-run model switches without losing error diagnostics or architectural decisions.
* **AST Symbol Mapper (`RepoMapper`):** Injects compact class/function signatures and numbered line ranges into every step prompt.
* **Workspace-Jailed `TerminalExecutor`:** Executes Zsh commands strictly inside `<project_dir>` with process-group timeout termination and destructive-command guardrails.

### 5. Multimodal & Multi-Format Chatbot Mode
* Interactive and one-shot CLI (`--sources`, `--option`, `--format`, `--out-dir`, `--prompt`) supporting PDFs, DOCX, PPTX, XLSX, images, web URLs, and structured file exports.
* Supports `/build` directly inside an interactive chat session to hand off preloaded sources to the autonomous DP builder.
* Preserves 100% backward compatibility for Python callers importing `from broutegpt import FreeLLMRouter`.

---

## Architecture Overview

```text
broutegpt/
├── broutegpt.py                     # Unified CLI entrypoint (Chatbot + Autonomous Agent) & FreeLLMRouter
├── broutegpt_agent/
│   ├── __init__.py                  # Public package exports
│   ├── cli_runner.py                # CLI orchestrator & shadow-Git forensics commands
│   ├── dp_orchestrator.py           # Zero-question DP state machine, verification gates & regression guard
│   ├── router_adapter.py            # CloudFirstRouter (tiered cloud escalation + last-resort Ollama)
│   ├── checkpoint_manager.py        # Hidden shadow-Git snapshots, main-Git DP tags & time-travel rollback
│   ├── code_editor.py               # Strict JSON envelope parser, SEARCH/REPLACE patcher & static linter
│   ├── repo_map.py                  # Python AST symbol extractor & numbered file reader
│   ├── event_stream.py              # Append-only JSONL event log
│   ├── condenser.py                 # Token-budgeted rolling context condenser
│   ├── input_inspector.py           # Input file staging into sample_inputs/ & schema extraction
│   └── terminal_executor.py         # Workspace-jailed Zsh runner with timeout & safety blocklist
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Python 3.12-slim + git + zsh container image
├── docker-compose.yml               # Compose service with runtime .env injection & workspace mount
├── .gitignore                       # Blocks .env, caches, and generated workspaces from Git
└── .dockerignore                    # Blocks .env and local state from Docker build context
```

---

## Installation & API Key Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bhagwatigb/broutegpt.git
cd broutegpt
```

### 2. Configure `.env` (Never Committed to Git or Docker)

Create a `.env` file in the root of `broutegpt` with whichever free API keys you have available (you do not need all of them—`broutegpt` automatically discovers active models for every key present):

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NVIDIA_API_KEY=your_nvidia_nim_api_key
HF_TOKEN=your_huggingface_token
OLLAMA_HOST=http://localhost:11434
```

> **Security Note:** `.env` is explicitly excluded in both `.gitignore` and `.dockerignore`. Your keys stay on your local machine and are injected into Docker containers strictly at runtime via `--env-file`.

### 3. Local Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Docker Setup

```bash
touch free_models_cache.json
docker build -t broutegpt:latest .
```

---

## Usage Guide

### A. Autonomous Project Building (Local)

Build a complete, tested software suite from a specification and input files:

```bash
python3 broutegpt.py \
  --build "Build a Python CLI tool analyze.py that accepts <fasta_path> <metadata_csv_path> <output_json_path>, joins sequences with condition on seq_id, computes length and GC% per sequence plus mean GC per condition, and writes valid JSON." \
  --inputs /path/to/sequences.fasta /path/to/samples.csv \
  --project-dir ./workspace/fasta_analyzer
```

### B. Autonomous Project Building (Docker)

```bash
mkdir -p ./workspace && \
docker run --rm \
  --env-file ./.env \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/workspace:/workspace" \
  -v "/path/to/your_data:/inputs:ro" \
  broutegpt:latest \
  --build "Analyze the staged input files and generate a summary report CLI" \
  --inputs /inputs \
  --project-dir /workspace/my_project
```

### C. Shadow-Git Forensics & Time-Travel Rollback

Every autonomous build records checkpoints in `<project_dir>/.broutegpt/`:

```bash
# View chronological timeline of steps, models used, and verified DPs
python3 broutegpt.py --timeline --project-dir ./workspace/fasta_analyzer

# Inspect the exact unified diff and model reasoning for Step #4
python3 broutegpt.py --inspect-step 4 --project-dir ./workspace/fasta_analyzer

# Rewind workspace files and agent memory back to Step #3, then resume building
python3 broutegpt.py --rollback 3 --project-dir ./workspace/fasta_analyzer
python3 broutegpt.py --resume --project-dir ./workspace/fasta_analyzer
```

*(In Docker, prefix the same flags with `docker run --rm -v "$(pwd)/workspace:/workspace" broutegpt:latest` and use `/workspace/...` paths.)*

### D. Multimodal Chatbot Mode

```bash
# Interactive chat session
python3 broutegpt.py

# Interactive chat via Docker Compose
docker compose run --rm broutegpt

# One-shot prompt with source files and structured export
python3 broutegpt.py \
  --sources report.pdf metrics.xlsx \
  --format md \
  --prompt "Summarize key findings and anomalies across both files."
```

Inside interactive chat mode, type `/build` at any time to launch the autonomous DP builder using your currently loaded `--sources`.

---

## CLI Reference

| Flag | Mode | Description |
| :--- | :--- | :--- |
| `--build "<goal>"` | Agent | Run the autonomous zero-question DP project builder for the given specification. |
| `--inputs <paths...>` | Agent | Input files or directories to copy into `<project_dir>/sample_inputs/` and inspect. |
| `--project-dir <dir>` | Agent / Forensics | Target workspace directory (default: `./broutegpt_project`). |
| `--max-dps <int>` | Agent | Maximum Decision Points to plan (default: `6`). |
| `--max-steps <int>` | Agent | Maximum build/debug actions per DP (default: `14`). |
| `--resume` | Agent | Resume an interrupted build or continue after `--rollback`. |
| `--timeline` | Forensics | Display shadow-Git checkpoints, model attribution, and verified DP milestones. |
| `--inspect-step <id>` | Forensics | Print unified Git diff and action details for a specific checkpoint step. |
| `--rollback <id>` | Forensics | Restore workspace files and rewind event context to a prior checkpoint step. |
| `--no-context-rollback` | Forensics | When used with `--rollback`, restore files only while retaining error history. |
| `--sources <items...>` | Chatbot | Preload PDFs, Office docs, images, folders, or URLs into chat context. |
| `--option <1-10\|id>` | Both | Preselect routing preset (`1`-`10`) or explicit `provider:model_id`. |
| `--format <fmt>` | Chatbot | Target export format (`auto`, `txt`, `md`, `json`, `csv`, `html`, `pdf`, `docx`, `xlsx`, `pptx`). |
| `--out-dir <dir>` | Chatbot | Directory for exported chatbot artifacts (default: `chatbot_outputs`). |
| `--prompt "<text>"` | Chatbot | Execute a single non-interactive prompt and exit. |
| `--refresh` | Both | Force refresh the cached list of available free models (`free_models_cache.json`). |

---

## Generated Project Layout

When an autonomous build finishes, `<project_dir>` contains:

```text
<project_dir>/
├── .git/                         # Main Git repo with clean `dp-<k>-verified` tags
├── .broutegpt/
│   ├── dp_plan.json              # Master Decision-Point blueprint & status
│   ├── build_report.json         # Entrypoint, verified DP summary & total runtime
│   ├── events.jsonl              # Full append-only audit log of actions & observations
│   ├── checkpoints.json          # Step-to-SHA forensic index
│   └── shadow_git/               # Hidden shadow Git repository for fine-grained diffs
├── sample_inputs/                # Staged copies of --inputs files
├── <entrypoint>.py               # Primary generated script/CLI
└── verify_dp1.py ...             # Standalone verification suites for every DP
```

---

## Python Library Usage (Backward Compatible)

```python
from broutegpt import FreeLLMRouter

router = FreeLLMRouter()
reply = router.query("Explain how GC skew is computed in bacterial genomes.")
print(reply)
```