import json
import sys
import subprocess
import pytest
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "optional-skills" / "mlops" / "effgen-orchestrator"
_RUN_SCRIPT = str(_SKILL_ROOT / "scripts" / "run_effgen_task.py")

_TIMEOUT = 60  # seconds — allow for optional slow model initialisation


def _run_task(config: dict) -> dict:
    result = subprocess.run(
        [sys.executable, _RUN_SCRIPT, "--config", json.dumps(config)],
        capture_output=True, text=True, timeout=_TIMEOUT,
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
