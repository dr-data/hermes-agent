"""
Integration tests for the effgen-orchestrator optional skill.

Organised in three sections:

1. **Local effGen** — uses the local effGen library (HuggingFace backend).
   Automatically skipped when effGen is not installed so CI stays green.

2. **OpenRouter — per-model smoke tests** — one test per model, simple tasks.
   Automatically skipped when OPENROUTER_API_KEY / OPENROUTER_API is absent.

3. **OpenRouter — usability / real-world tests** — richer task-type coverage
   across all three models: routing verification, QA, coding, summarisation,
   chain-of-thought reasoning, and output-structure checks.
   Same skip logic as section 2.

All tests call ``run_effgen_task.py`` as a subprocess to mirror the exact
execution path that Hermes uses when it invokes this skill via the terminal tool.
"""

import json
import os
import sys
import subprocess
import pytest
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "optional-skills" / "mlops" / "effgen-orchestrator"
_RUN_SCRIPT = str(_SKILL_ROOT / "scripts" / "run_effgen_task.py")

# Allow more time for live API calls; local effGen may need even more.
_TIMEOUT = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_task(config: dict, env: dict = None) -> dict:
    """Invoke run_effgen_task.py as a subprocess and return the parsed JSON."""
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


def _assert_valid_result_schema(output: dict):
    """Verify the result conforms to the standard result-schema.json shape."""
    required_keys = {
        "mode_selected", "summary", "output", "artifacts",
        "errors", "metrics", "next_action",
    }
    missing = required_keys - set(output.keys())
    assert not missing, f"Missing keys in result: {missing}\nFull output: {output}"
    assert isinstance(output["errors"], list), "errors must be a list"
    assert isinstance(output["artifacts"], list), "artifacts must be a list"


def _skip_if_not_installed(output: dict):
    """Skip gracefully when effGen is not installed in the environment."""
    if output.get("mode_selected") == "error":
        err_msg = (output.get("errors") or [{}])[0].get("message", "")
        if "not installed" in err_msg:
            pytest.skip("effGen not installed; skipping real SLM execution")


def _skip_if_no_openrouter_key():
    """Skip when neither OPENROUTER_API_KEY nor OPENROUTER_API is set."""
    has_key = (
        os.environ.get("OPENROUTER_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API", "").strip()
    )
    if not has_key:
        pytest.skip("OPENROUTER_API_KEY / OPENROUTER_API not set; skipping OpenRouter test")


def _is_temporary_openrouter_upstream_429(error: dict) -> bool:
    """Return True for transient upstream 429 errors that should be skipped in CI."""
    message = str(error.get("message", "")).lower()
    details = str(error.get("details", "")).lower()
    return (
        "http 429" in message
        and (
            "temporarily rate-limited upstream" in details
            or "please retry shortly" in details
        )
    )


def _skip_if_temporary_openrouter_upstream_429(output: dict, model: str):
    """Skip flaky assertions when OpenRouter reports transient upstream throttling."""
    for err in output.get("errors", []):
        if isinstance(err, dict) and _is_temporary_openrouter_upstream_429(err):
            pytest.skip(
                f"OpenRouter upstream is temporarily rate-limited for {model}; skipping"
            )


def _openrouter_config(model: str, goal: str, task_type: str = "minimal", **extra) -> dict:
    """Return a minimal OpenRouter config dict with sensible defaults."""
    cfg = {
        "task_type": task_type,
        "user_goal": goal,
        "api_backend": "openrouter",
        "openrouter_model": model,
        "need_decomposition": False,
        "timeout": 45,
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Section 1 — Local effGen integration (HuggingFace backend)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_effgen_orchestrator_integration_small_model():
    """Qwen2.5-1.5B-Instruct via local effGen; skipped if effGen absent."""
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
    """liquid-ai/LFM2-1B via local effGen; skipped if effGen absent."""
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
    """google/gemma-3-1b-it via local effGen; skipped if effGen absent."""
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
# Section 2 — OpenRouter per-model smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_effgen_openrouter_lfm25_instruct():
    """
    Smoke test: liquid/lfm-2.5-1.2b-instruct:free via OpenRouter.

    Validates that the OpenRouter backend is reachable and the model returns
    a correct answer to a simple arithmetic question.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-instruct:free",
        "What is 3 multiplied by 7? Respond with only the number.",
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["mode_selected"].startswith("openrouter:"), (
        f"Expected openrouter mode, got: {output['mode_selected']!r}"
    )
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip()
    assert "21" in str(output["output"]), (
        f"Expected '21' in output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_effgen_openrouter_lfm25_thinking():
    """
    Smoke test: liquid/lfm-2.5-1.2b-thinking:free via OpenRouter.

    Validates the 'thinking' variant of the LFM-2.5 model.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-thinking:free",
        "What is the capital of Germany? Answer in one word.",
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["mode_selected"].startswith("openrouter:")
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip()
    assert "berlin" in str(output["output"]).lower(), (
        f"Expected 'berlin' in output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_effgen_openrouter_gemma3n_e2b():
    """
    Smoke test: google/gemma-3n-e2b-it:free via OpenRouter.

    Validates Google's Gemma 3n Edge 2B model on a basic arithmetic task.
    """
    _skip_if_no_openrouter_key()
    model = "google/gemma-3n-e2b-it:free"
    config = _openrouter_config(
        model,
        "What is 6 plus 9? Respond with only the number.",
    )
    output = _run_task(config)
    _skip_if_temporary_openrouter_upstream_429(output, model)
    _assert_valid_result_schema(output)
    assert output["mode_selected"].startswith("openrouter:")
    assert output["errors"] == [], f"Unexpected errors: {output['errors']}"
    assert output["output"] is not None and str(output["output"]).strip()
    assert "15" in str(output["output"]), (
        f"Expected '15' in output, got: {output['output']!r}"
    )


# ---------------------------------------------------------------------------
# Section 3 — OpenRouter usability / real-world tests
# ---------------------------------------------------------------------------
# These tests exercise the full usability of the effgen-orchestrator skill as
# Hermes would invoke it: different task_type values trigger different routing
# modes, and diverse prompts validate that the models are actually useful for
# typical Hermes user requests.
# ---------------------------------------------------------------------------

# ---- Result-schema and metric tests ----------------------------------------

@pytest.mark.integration
def test_openrouter_result_contains_token_metrics():
    """
    The OpenRouter backend must return prompt/completion/total token counts in
    ``metrics`` so Hermes can track usage.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-instruct:free",
        "Say 'hello'.",
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    metrics = output.get("metrics", {})
    # OpenRouter always returns usage; at least total_tokens must be present
    assert metrics.get("total_tokens") is not None, (
        f"Expected token usage in metrics, got: {metrics}"
    )
    assert int(metrics["total_tokens"]) > 0, "total_tokens must be positive"


@pytest.mark.integration
def test_openrouter_result_mode_label_contains_model_name():
    """
    ``mode_selected`` must be ``openrouter:<model_id>`` so callers can identify
    which model actually handled the request.
    """
    _skip_if_no_openrouter_key()
    model = "google/gemma-3n-e2b-it:free"
    config = _openrouter_config(model, "Say 'ok'.")
    output = _run_task(config)
    _skip_if_temporary_openrouter_upstream_429(output, model)
    _assert_valid_result_schema(output)
    assert output["mode_selected"] == f"openrouter:{model}", (
        f"Unexpected mode_selected: {output['mode_selected']!r}"
    )


# ---- Routing mode tests (task_type drives effgen preset selection) ----------

@pytest.mark.integration
def test_openrouter_routing_math_task_type():
    """
    task_type='math' should route to preset_math in the mode-selection logic.
    When using OpenRouter, the backend is always 'openrouter:<model>' regardless
    of preset, but the config round-trips correctly without error.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-instruct:free",
        "What is the square root of 144? Answer with only the number.",
        task_type="math",
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    assert output["output"] is not None and str(output["output"]).strip()
    assert "12" in str(output["output"]), (
        f"Expected '12' in output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_openrouter_routing_code_task_type():
    """
    task_type='code' verifies the skill handles a coding request end-to-end
    via OpenRouter, returning syntactically plausible Python.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-instruct:free",
        (
            "Write a Python function called `add` that takes two integers and "
            "returns their sum. Output only the function definition, no explanation."
        ),
        task_type="code",
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    result_text = str(output["output"]).lower()
    assert "def add" in result_text, (
        f"Expected 'def add' in code output, got: {output['output']!r}"
    )
    assert "return" in result_text, (
        f"Expected 'return' in code output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_openrouter_routing_qa_task_type_gemma():
    """
    task_type='qa' with google/gemma-3n-e2b-it:free — factual QA test.
    Validates that Gemma 3n correctly answers a knowledge question.
    """
    _skip_if_no_openrouter_key()
    model = "google/gemma-3n-e2b-it:free"
    config = _openrouter_config(
        model,
        "How many planets are in the solar system? Answer with only the number.",
        task_type="qa",
    )
    output = _run_task(config)
    _skip_if_temporary_openrouter_upstream_429(output, model)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    assert "8" in str(output["output"]), (
        f"Expected '8' in output, got: {output['output']!r}"
    )


@pytest.mark.integration
def test_openrouter_routing_research_task_type_thinking():
    """
    task_type='research' with liquid/lfm-2.5-1.2b-thinking:free.
    Tests that the thinking model handles a structured research-style question.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-thinking:free",
        (
            "In one sentence, explain what large language models are used for "
            "in modern AI assistants."
        ),
        task_type="research",
        timeout=60,
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    result_text = str(output["output"]).strip()
    assert len(result_text) > 20, (
        f"Expected a meaningful research response (>20 chars), got: {result_text!r}"
    )


# ---- Summarisation test -------------------------------------------------------

@pytest.mark.integration
def test_openrouter_summarisation_gemma():
    """
    Tests that google/gemma-3n-e2b-it:free can summarise a short paragraph.
    This validates the model's text-compression capability via the skill.
    """
    _skip_if_no_openrouter_key()
    model = "google/gemma-3n-e2b-it:free"
    paragraph = (
        "The Python programming language was created by Guido van Rossum and first "
        "released in 1991. It emphasises code readability and simplicity, making it "
        "popular for web development, data science, artificial intelligence, and "
        "automation. Python uses dynamic typing and garbage collection, and supports "
        "multiple programming paradigms including procedural, object-oriented, and "
        "functional programming."
    )
    config = _openrouter_config(
        model,
        f"Summarise the following in exactly one sentence:\n\n{paragraph}",
        task_type="general",
        timeout=60,
    )
    output = _run_task(config)
    _skip_if_temporary_openrouter_upstream_429(output, model)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    result_text = str(output["output"]).strip()
    assert len(result_text) > 10, "Expected a non-trivial summary"
    # The summary should mention Python
    assert "python" in result_text.lower(), (
        f"Expected 'python' in summary, got: {result_text!r}"
    )


# ---- Chain-of-thought / reasoning test ----------------------------------------

@pytest.mark.integration
def test_openrouter_chain_of_thought_lfm25_thinking():
    """
    Tests that liquid/lfm-2.5-1.2b-thinking:free produces a step-by-step
    reasoning response for a multi-step arithmetic word problem.
    """
    _skip_if_no_openrouter_key()
    config = _openrouter_config(
        "liquid/lfm-2.5-1.2b-thinking:free",
        (
            "Alice has 5 apples. She gives 2 to Bob and then buys 4 more. "
            "How many apples does Alice have now? Show your working and give the final answer."
        ),
        task_type="math",
        timeout=60,
    )
    output = _run_task(config)
    _assert_valid_result_schema(output)
    assert output["errors"] == []
    result_text = str(output["output"])
    assert "7" in result_text, (
        f"Expected answer '7' in reasoning output, got: {result_text!r}"
    )


# ---- Cross-model consistency test ---------------------------------------------

@pytest.mark.integration
def test_openrouter_all_three_models_answer_same_question():
    """
    Sends the same factual question to all three OpenRouter models and asserts
    each returns the correct answer.  This validates that the skill works
    identically across all three tested models.
    """
    _skip_if_no_openrouter_key()
    models = [
        "liquid/lfm-2.5-1.2b-instruct:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
        "google/gemma-3n-e2b-it:free",
    ]
    question = "What is 10 divided by 2? Answer with only the number."
    failures = []
    for model in models:
        config = _openrouter_config(model, question)
        output = _run_task(config)
        _skip_if_temporary_openrouter_upstream_429(output, model)
        _assert_valid_result_schema(output)
        if output["errors"]:
            failures.append(f"{model}: errors={output['errors']}")
        elif "5" not in str(output["output"]):
            failures.append(f"{model}: expected '5', got {output['output']!r}")
    assert not failures, "Some models failed:\n" + "\n".join(failures)


def test_is_temporary_openrouter_upstream_429_detection():
    error = {
        "message": "OpenRouter API returned HTTP 429.",
        "details": (
            '{"error":{"message":"Provider returned error","code":429,'
            '"metadata":{"raw":"model is temporarily rate-limited upstream. '
            'Please retry shortly"}}}'
        ),
    }
    assert _is_temporary_openrouter_upstream_429(error) is True


def test_is_temporary_openrouter_upstream_429_not_matched_for_other_429():
    error = {
        "message": "OpenRouter API returned HTTP 429.",
        "details": '{"error":{"message":"Insufficient credits","code":429}}',
    }
    assert _is_temporary_openrouter_upstream_429(error) is False


# ---- Error path (always-on, no real API call) ---------------------------------

def test_effgen_openrouter_missing_key_error():
    """
    When OPENROUTER_API_KEY is absent the script must return a structured error,
    not crash.  Forces the missing-key path regardless of env state.

    Not marked @pytest.mark.integration — always runs in CI without a key.
    """
    config = {
        "task_type": "minimal",
        "user_goal": "Hello",
        "api_backend": "openrouter",
        "openrouter_model": "liquid/lfm-2.5-1.2b-instruct:free",
    }
    output = _run_task(config, env={"OPENROUTER_API_KEY": "", "OPENROUTER_API": ""})
    _assert_valid_result_schema(output)
    assert output["mode_selected"] == "error"
    assert any("OPENROUTER_API_KEY" in e.get("message", "") for e in output["errors"])
