import json
import subprocess
import pytest
import os

@pytest.mark.integration
def test_effgen_orchestrator_integration_small_model():
    """
    Integration test to verify how the effgen-orchestrator skill works with a real small language model.
    This test runs the actual scripts/run_effgen_task.py with a <2B parameter model.
    Requires `effgen` to be installed and available in the environment.
    """
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/run_effgen_task.py"

    config = {
        "task_type": "minimal",
        "user_goal": "What is 2 + 2? Answer with just the number.",
        "model_name": "liquid/lfm-2.5-1.2b-instruct:free", # <2B parameter model suitable for SLM testing
        "quantization": "4bit",
        "preset": "minimal",
        "need_decomposition": False
    }

    # Pass the provided OpenRouter API key to the environment for remote model testing if supported
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "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"
    env["OPENAI_API_KEY"] = "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"
    env["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    result = subprocess.run(
        ["python3", script_path, json.dumps(config)],
        capture_output=True,
        text=True,
        env=env
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)

    # If effgen is not installed in the test environment, the script safely returns an error payload
    if output["mode_selected"] == "error":
        assert "effGen library not installed" in output["errors"][0]["message"]
        pytest.skip("effGen not installed, skipping real SLM execution")

    assert output["mode_selected"] == "preset_minimal"
    assert "4" in str(output["output"]).lower()
