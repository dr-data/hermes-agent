import json
import os
import sys
import subprocess
import pytest
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "optional-skills" / "mlops" / "effgen-orchestrator"
_RUN_SCRIPT = str(_SKILL_ROOT / "scripts" / "run_effgen_task.py")

_TIMEOUT = 60  # seconds — allow for optional slow model initialization


def _run_task(config: dict, env: dict = None) -> dict:
    run_env = {**os.environ}
    if env:
        run_env.update(env)
    result = subprocess.run(
        [sys.executable, _RUN_SCRIPT, "--config", json.dumps(config)],
        capture_output=True, text=True, timeout=_TIMEOUT,
        env=run_env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _skip_if_not_installed(output: dict):
    """Skip the test gracefully when effGen is not present in the environment."""
    if output.get("mode_selected") == "error":
        err_msg = (output.get("errors") or [{}])[0].get("message", "")
        if "not installed" in err_msg:
            pytest.skip("effGen not installed; skipping real SLM execution")


@pytest.mark.integration
def test_effgen_orchestrator_integration_small_model():
    """
    Integration test: Qwen2.5-1.5B-Instruct (<2 B) via effGen.

    Skipped automatically when effGen is not installed so CI remains green.
    """
    config = {
        "task_type": "minimal",
        "user_goal": "What is 2 + 2? Answer with just the number.",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": "4bit",
        "preset": "minimal",
        "need_decomposition": False,
    }
    output = _run_task(config)
    _skip_if_not_installed(output)
    assert output["mode_selected"] == "preset_minimal"
    assert "4" in str(output["output"])


@pytest.mark.integration
def test_effgen_orchestrator_integration_lfm2():
    """
    Integration test: LFM2-1B (Liquid Foundation Model, <2 B) via effGen.

    Validates that the skill routes and executes correctly with the LFM2-1B
    HuggingFace model.  Skipped when effGen is not installed.
    """
    config = {
        "task_type": "minimal",
        "user_goal": "What is the capital of France? Answer in one word.",
        "model_name": "liquid-ai/LFM2-1B",
        "quantization": "4bit",
        "preset": "minimal",
        "need_decomposition": False,
    }
    output = _run_task(config)
    _skip_if_not_installed(output)
    assert output["mode_selected"] == "preset_minimal"
    assert output["errors"] == []


@pytest.mark.integration
def test_effgen_orchestrator_integration_gemma():
    """
    Integration test: Gemma-3 1B (<2 B) via effGen.

    Validates that the skill routes and executes correctly with Google's
    Gemma-3 1B instruction-tuned model.  Skipped when effGen is not installed.
    """
    config = {
        "task_type": "qa",
        "user_goal": "Is Python a compiled or interpreted language? Answer in one word.",
        "model_name": "google/gemma-3-1b-it",
        "quantization": "4bit",
        "preset": "minimal",
        "need_decomposition": False,
    }
    output = _run_task(config)
    _skip_if_not_installed(output)
    assert output["mode_selected"] == "preset_minimal"
    assert output["errors"] == []


# ---------------------------------------------------------------------------
# OpenRouter integration tests — liquid/lfm-2.5-1.2b-instruct and thinking
# ---------------------------------------------------------------------------
# These tests call the OpenRouter cloud API via run_effgen_task.py's openrouter
# backend.  They require OPENROUTER_API_KEY to be set in the environment and
# are automatically skipped when the variable is absent so CI stays green.
# ---------------------------------------------------------------------------

def _skip_if_no_openrouter_key():
    """Skip the test when neither OPENROUTER_API_KEY nor OPENROUTER_API is configured."""
    has_key = (
        os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API", "").strip()
    )
    if not has_key:
        pytest.skip("OPENROUTER_API_KEY / OPENROUTER_API not set; skipping OpenRouter integration test")


@pytest.mark.integration
def test_effgen_openrouter_lfm25_instruct():
    """
    Integration test: liquid/lfm-2.5-1.2b-instruct via OpenRouter.

    Uses the effgen-orchestrator OpenRouter backend (no local GPU required).
    Requires OPENROUTER_API_KEY to be set in the environment.
    """
    _skip_if_no_openrouter_key()
    config = {
        "task_type": "minimal",
        "user_goal": "What is 3 multiplied by 7? Respond with only the number.",
        "api_backend": "openrouter",
        "openrouter_model": "liquid/lfm-2.5-1.2b-instruct:free",
        "preset": "minimal",
        "need_decomposition": False,
        "timeout": 30,
    }
    output = _run_task(config)
    assert output.get("mode_selected", "").startswith("openrouter:"), (
        f"Expected openrouter mode, got: {output}"
    )
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip(), (
        "Expected non-empty output from the model"
    )
    # The model should respond with the correct answer (21)
    assert "21" in str(output["output"]), (
        f"Expected '21' in model output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_effgen_openrouter_lfm25_thinking():
    """
    Integration test: liquid/lfm-2.5-1.2b-thinking via OpenRouter.

    Uses the effgen-orchestrator OpenRouter backend with the 'thinking' variant
    of the LFM-2.5-1.2B model.  Requires OPENROUTER_API_KEY to be set.
    """
    _skip_if_no_openrouter_key()
    config = {
        "task_type": "minimal",
        "user_goal": "What is the capital of Germany? Answer in one word.",
        "api_backend": "openrouter",
        "openrouter_model": "liquid/lfm-2.5-1.2b-thinking:free",
        "preset": "minimal",
        "need_decomposition": False,
        "timeout": 30,
    }
    output = _run_task(config)
    assert output.get("mode_selected", "").startswith("openrouter:"), (
        f"Expected openrouter mode, got: {output}"
    )
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip(), (
        "Expected non-empty output from the model"
    )
    assert "berlin" in str(output["output"]).lower(), (
        f"Expected 'berlin' in model output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_effgen_openrouter_gemma3n_e2b():
    """
    Integration test: google/gemma-3n-e2b-it via OpenRouter.

    Uses the effgen-orchestrator OpenRouter backend with Google's Gemma 3n
    Edge 2B model.  Requires OPENROUTER_API_KEY to be set.
    """
    _skip_if_no_openrouter_key()
    config = {
        "task_type": "minimal",
        "user_goal": "What is 6 plus 9? Respond with only the number.",
        "api_backend": "openrouter",
        "openrouter_model": "google/gemma-3n-e2b-it:free",
        "preset": "minimal",
        "need_decomposition": False,
        "timeout": 30,
    }
    output = _run_task(config)
    assert output.get("mode_selected", "").startswith("openrouter:"), (
        f"Expected openrouter mode, got: {output}"
    )
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip(), (
        "Expected non-empty output from the model"
    )
    assert "15" in str(output["output"]), (
        f"Expected '15' in model output, got: {output['output']!r}"
    )


def test_effgen_openrouter_missing_key_error():
    """
    When OPENROUTER_API_KEY is absent the script must return a structured error,
    not crash.  This test forces the missing-key path regardless of env state.

    Not marked @pytest.mark.integration because it does not call the real API
    and should always run in CI.
    """
    config = {
        "task_type": "minimal",
        "user_goal": "Hello",
        "api_backend": "openrouter",
        "openrouter_model": "liquid/lfm-2.5-1.2b-instruct:free",
    }
    # Explicitly strip both key names so the error path is always exercised
    output = _run_task(config, env={"OPENROUTER_API_KEY": "", "OPENROUTER_API": ""})
    assert output["mode_selected"] == "error"
    assert any("OPENROUTER_API_KEY" in e.get("message", "") for e in output["errors"])
