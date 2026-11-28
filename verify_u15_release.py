import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

os.environ["BROUTEGPT_NO_SANDBOX"] = "1"


def test_modular_imports_and_pyproject() -> None:
    repo_root = Path(__file__).resolve().parent
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.is_file(), "Missing pyproject.toml"
    parsed_toml = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert parsed_toml["project"]["name"] == "broutegpt"
    assert parsed_toml["project"]["scripts"]["broutegpt"] == "broutegpt_core.cli:main"

    for mod_name in (
        "broutegpt",
        "broutegpt_core",
        "broutegpt_core.config",
        "broutegpt_core.sources",
        "broutegpt_core.router",
        "broutegpt_core.exporters",
        "broutegpt_core.cli",
        "broutegpt_agent",
        "broutegpt_agent.mcp_bridge",
    ):
        __import__(mod_name)

    import broutegpt
    import broutegpt_agent

    assert hasattr(broutegpt, "FreeLLMRouter")
    assert hasattr(broutegpt, "load_all_api_keys")
    assert hasattr(broutegpt, "run_doctor")
    assert hasattr(broutegpt, "run_chatbot_cli")
    assert hasattr(broutegpt, "main")
    assert hasattr(broutegpt_agent, "MCPRegistry")
    assert hasattr(broutegpt_agent, "DPOrchestrator")
    print("[PASS] 1. Modular package structure (broutegpt_core/, broutegpt.py shim, pyproject.toml)")


def test_cumulative_regression_suite() -> None:
    repo_root = Path(__file__).resolve().parent
    scripts = sorted(
        repo_root.glob("verify_u*.py"),
        key=lambda p: int(p.name.split("_")[1][1:]),
    )
    prior_scripts = [
        p for p in scripts
        if p.name not in ("verify_u1_baseline.py", "verify_u15_release.py")
    ]
    assert len(prior_scripts) >= 10, f"Expected full u1..u14 verification scripts, found {len(prior_scripts)}"

    for script in prior_scripts:
        res = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            env={**os.environ, "BROUTEGPT_NO_SANDBOX": "1"},
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, (
            f"Regression failure in {script.name} (exit={res.returncode}):\n"
            f"STDOUT:\n{res.stdout[-1500:]}\nSTDERR:\n{res.stderr[-1500:]}"
        )
        print(f"  [OK] {script.name}")
    print(f"[PASS] 2. All {len(prior_scripts)} cumulative upgrade verification scripts passed")


def test_live_smoke_build_and_cli() -> None:
    repo_root = Path(__file__).resolve().parent
    broutegpt_py = repo_root / "broutegpt.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        res_doc = subprocess.run(
            [sys.executable, str(broutegpt_py), "--doctor", "--json"],
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": str(repo_root)},
            capture_output=True,
            text=True,
            check=True,
        )
        diag = json.loads(res_doc.stdout)
        assert diag["healthy"] is True

        # Check if any live provider key or local Ollama is available for a real live smoke build
        prod_env = Path.home() / "broutegpt" / ".env"
        if prod_env.is_file():
            from broutegpt_core.config import _load_env_file

            _load_env_file(str(prod_env))

        has_live_backend = diag["ollama"]["reachable"] or any(
            bool(os.environ.get(k, "").strip()) and not os.environ.get(k, "").startswith("your_")
            for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "HF_API_KEY")
        )

        if has_live_backend:
            smoke_dir = tmp_path / "live_smoke_ws"
            res_build = subprocess.run(
                [
                    sys.executable,
                    str(broutegpt_py),
                    "--build",
                    "Create math_ops.py with add(a, b) and multiply(a, b) plus verify_dp1.py testing both. Use 1 DP only.",
                    "--project-dir",
                    str(smoke_dir),
                    "--max-dps",
                    "1",
                    "--max-steps",
                    "6",
                    "--bare",
                    "--json",
                ],
                cwd=str(repo_root),
                env={**os.environ, "PYTHONPATH": str(repo_root), "BROUTEGPT_NO_SANDBOX": "1"},
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert res_build.returncode == 0, (
                f"Live smoke build failed (exit={res_build.returncode}):\n"
                f"STDOUT:\n{res_build.stdout[-1500:]}\nSTDERR:\n{res_build.stderr[-1500:]}"
            )
            rep = json.loads(res_build.stdout)
            assert rep["total_dps"] == 1
            assert (smoke_dir / "verify_dp1.py").is_file()
            print(f"[PASS] 3. Live smoke build succeeded ({rep['llm_calls']} LLM call(s), {rep['total_tokens']} tokens)")
        else:
            print("[PASS] 3. CLI --doctor verified (no live API key/Ollama active; skipped network smoke build)")


if __name__ == "__main__":
    test_modular_imports_and_pyproject()
    test_cumulative_regression_suite()
    test_live_smoke_build_and_cli()
    print("\nALL DP 15 MODULAR REFACTOR & CUMULATIVE RELEASE CHECKS PASSED!")

