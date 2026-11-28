import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["BROUTEGPT_NO_SANDBOX"] = "1"


def _write_fake_ollama_and_cache(tmp_dir: Path) -> Path:
    """Creates a local mock cache so broutegpt CLI runs offline without network calls."""
    cache_file = tmp_dir / ".free_models_cache.json"
    cache_payload = {
        "models_by_option": {
            "1": [
                {
                    "id": "mock-local-7b",
                    "provider": "ollama",
                    "input_limit": 32768,
                    "output_limit": 8192,
                    "vision": False,
                    "chat": True,
                    "est_tps": 120.0,
                }
            ],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
        },
        "role_config": {
            "image_transcription": [],
            "image_generation": [],
            "audio_transcription": [],
            "fast_triage": [{"provider": "ollama", "id": "mock-local-7b"}],
            "deep_synthesis": [{"provider": "ollama", "id": "mock-local-7b"}],
        },
    }
    cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    return cache_file


def _start_mock_ollama_server():
    import http.server
    import socketserver
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/api/tags":
                body = json.dumps({"models": [{"name": "mock-local-7b"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            req = json.loads(raw or "{}")
            msgs = req.get("messages") or []
            last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
            sys_msg = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")

            if "Decision-Point (DP) execution plan" in sys_msg or "Return ONLY valid JSON in this exact schema" in last_user:
                reply_text = json.dumps({
                    "project_name": "planned_demo_cli",
                    "overview": "Read-only planned CLI utility.",
                    "entrypoint": "app.py",
                    "dps": [
                        {
                            "id": 1,
                            "title": "Core Parser",
                            "depends_on": [],
                            "description": "Implement app.py",
                            "deliverables": ["app.py", "verify_dp1.py"],
                            "verification_criteria": "assert True",
                        }
                    ],
                })
            elif "sizing the output token budget" in last_user:
                reply_text = json.dumps({
                    "estimated_output_tokens": 256,
                    "reasoning_depth": "light",
                    "rationale": "Short CLI test",
                })
            else:
                first_line = last_user.strip().splitlines()[-1] if last_user.strip() else "empty"
                reply_text = f"ECHO_OK: {first_line}"

            resp_obj = {
                "message": {"role": "assistant", "content": reply_text},
                "prompt_eval_count": 42,
                "eval_count": 18,
            }
            body = json.dumps(resp_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{port}"


def test_doctor_flag() -> None:
    repo_root = Path(__file__).resolve().parent
    broutegpt_py = repo_root / "broutegpt.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fake_ollama_and_cache(tmp_path)
        server, ollama_url = _start_mock_ollama_server()
        try:
            env = {
                **os.environ,
                "PYTHONPATH": str(repo_root),
                "OLLAMA_BASE_URL": ollama_url,
                "GEMINI_API_KEY": "AIzaFakeDoctorTestKey12345",
            }
            res_json = subprocess.run(
                [sys.executable, str(broutegpt_py), "--doctor", "--json"],
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            diag = json.loads(res_json.stdout)
            assert diag["healthy"] is True, f"Expected healthy doctor output: {diag}"
            assert diag["git"]["available"] is True
            assert diag["ollama"]["reachable"] is True
            assert diag["providers"]["GEMINI_API_KEY"]["configured"] is True
            assert "AIzaFakeDoctorTestKey12345" not in res_json.stdout, "Raw API key leaked in --doctor output!"

            res_txt = subprocess.run(
                [sys.executable, str(broutegpt_py), "--doctor"],
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            assert "broutegpt --doctor Environment Diagnostics" in res_txt.stdout
        finally:
            server.shutdown()
    print("[PASS] 1. --doctor environment, provider, Git, sandbox, and secret-redacted diagnostics")


def test_piping_print_and_json_output() -> None:
    repo_root = Path(__file__).resolve().parent
    broutegpt_py = repo_root / "broutegpt.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fake_ollama_and_cache(tmp_path)
        server, ollama_url = _start_mock_ollama_server()
        try:
            env = {
                **os.environ,
                "PYTHONPATH": str(repo_root),
                "OLLAMA_BASE_URL": ollama_url,
            }
            # 1) Unix piped stdin + -p + --print
            res_pipe = subprocess.run(
                [sys.executable, str(broutegpt_py), "-p", "Summarize line", "--print", "--option", "1"],
                cwd=str(tmp_path),
                env=env,
                input="hello from unix pipe\n",
                capture_output=True,
                text=True,
                check=True,
            )
            assert "ECHO_OK: Summarize line" in res_pipe.stdout, f"Unexpected piped output: {res_pipe.stdout}"
            assert "FREE LLM ROUTER" not in res_pipe.stdout, "Banner should be suppressed in --print mode"

            # 2) --output-format json emits pure parseable JSON on stdout
            res_json = subprocess.run(
                [sys.executable, str(broutegpt_py), "-p", "Return status", "--output-format", "json", "--option", "1"],
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(res_json.stdout)
            assert parsed["status"] == "ok"
            assert parsed["model"] == "ollama:mock-local-7b"
            assert "ECHO_OK: Return status" in parsed["response"]
        finally:
            server.shutdown()
    print("[PASS] 2. Unix stdin piping, -p/--print clean output, and --output-format json")


def test_plan_and_bare_flags() -> None:
    repo_root = Path(__file__).resolve().parent
    broutegpt_py = repo_root / "broutegpt.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fake_ollama_and_cache(tmp_path)
        proj_dir = tmp_path / "planned_ws"
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "AGENTS.md").write_text("SECRET_RULE_SHOULD_BE_SKIPPED_IN_BARE\n", encoding="utf-8")
        state_dir = proj_dir / ".broutegpt"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "MEMORY.md").write_text("SECRET_MEMORY_SHOULD_BE_SKIPPED_IN_BARE\n", encoding="utf-8")

        server, ollama_url = _start_mock_ollama_server()
        try:
            env = {
                **os.environ,
                "PYTHONPATH": str(repo_root),
                "OLLAMA_BASE_URL": ollama_url,
            }
            res_plan = subprocess.run(
                [
                    sys.executable,
                    str(broutegpt_py),
                    "--plan",
                    "--bare",
                    "--json",
                    "--build",
                    "Build a small calculator CLI",
                    "--project-dir",
                    str(proj_dir),
                    "--option",
                    "1",
                ],
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
            )
            assert res_plan.returncode == 0, f"--plan failed (exit={res_plan.returncode}):\nSTDOUT:\n{res_plan.stdout}\nSTDERR:\n{res_plan.stderr}"
            plan_data = json.loads(res_plan.stdout)
            assert plan_data["project_name"] == "planned_demo_cli"
            assert len(plan_data["dps"]) == 1
            assert (state_dir / "dp_plan.json").is_file()
            # --plan must NOT create app.py or verify_dp1.py
            assert not (proj_dir / "app.py").exists(), "--plan must not mutate workspace source files"
            assert not (proj_dir / "verify_dp1.py").exists(), "--plan must not execute DPs"

            # Verify --bare skipped AGENTS.md and MEMORY.md rules in events.jsonl
            events_txt = (state_dir / "events.jsonl").read_text(encoding="utf-8")
            assert "SECRET_RULE_SHOULD_BE_SKIPPED_IN_BARE" not in events_txt
            assert "SECRET_MEMORY_SHOULD_BE_SKIPPED_IN_BARE" not in events_txt
        finally:
            server.shutdown()
    print("[PASS] 3. --plan read-only planning and --bare CI rule/memory isolation")


if __name__ == "__main__":
    test_doctor_flag()
    test_piping_print_and_json_output()
    test_plan_and_bare_flags()
    print("\nALL DP 13 CLI UX, PIPING, BARE, PLAN, JSON & DOCTOR CHECKS PASSED!")


