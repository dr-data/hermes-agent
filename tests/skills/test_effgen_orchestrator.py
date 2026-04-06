import importlib.util
import json
import sys
import tempfile
import subprocess
from pathlib import Path

# Absolute paths derived from this file's location so tests work regardless
# of the working directory from which pytest is invoked.
_SKILL_ROOT = Path(__file__).resolve().parents[2] / "optional-skills" / "mlops" / "effgen-orchestrator"
_SELECT_SCRIPT = str(_SKILL_ROOT / "scripts" / "select_effgen_mode.py")
_RUN_SCRIPT = str(_SKILL_ROOT / "scripts" / "run_effgen_task.py")
_EXAMPLE_CONFIG = str(_SKILL_ROOT / "templates" / "task-config.example.json")

# Subprocess timeout (seconds) so tests never hang in CI.
_TIMEOUT = 30


def _load_run_module():
    """Import run_effgen_task as a module for direct unit testing."""
    spec = importlib.util.spec_from_file_location("run_effgen_task", _RUN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_effgen_mode():
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT, _EXAMPLE_CONFIG],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "preset_research"
    assert output["preset"] == "research"


def test_run_effgen_task_missing_deps():
    """Script must return a structured error JSON when effGen is not installed.

    The test is skipped when effGen happens to be present in the environment
    so that it does not trigger an unwanted (and potentially slow) model load.
    """
    try:
        import effgen  # noqa: F401
        import pytest
        pytest.skip("effGen is installed; missing-deps path cannot be exercised")
    except ImportError:
        pass

    result = subprocess.run(
        [sys.executable, _RUN_SCRIPT, _EXAMPLE_CONFIG],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "error"
    assert len(output["errors"]) > 0
    assert "effGen library not installed" in output["errors"][0]["message"]


def test_select_effgen_mode_multi_agent():
    config = {"task_type": "complex", "need_decomposition": True}
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input=json.dumps(config), capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "custom_multi_agent"
    assert output["need_decomposition"] is True


def test_select_effgen_mode_tool_pipeline():
    config = {"task_type": "general", "custom_tools": ["my_tool"]}
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input=json.dumps(config), capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "custom_tool_pipeline"
    assert output["need_custom_tools"] is True


def test_select_effgen_mode_memory():
    config = {"task_type": "chat", "need_memory": True}
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input=json.dumps(config), capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "memory_augmented_run"
    assert output["need_memory"] is True


def test_run_effgen_mock_installed():
    """Verify routing logic by monkey-patching effGen into the module at runtime."""
    mock_runner_script = f"""\
import json, sys, importlib.util

spec = importlib.util.spec_from_file_location("run_effgen_task", {_RUN_SCRIPT!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules["run_effgen_task"] = module

module.EFFGEN_INSTALLED = True

class MockModel:
    pass

module.load_model = lambda name, quantization: MockModel()
module.create_agent = True
module.run_preset_mode = lambda p, m, g: module.emit_result("preset_" + p, "success", "preset output", [])
module.run_custom_multi_agent = lambda c, m, g: module.emit_result("custom_multi_agent", "success", "multi agent output", [])

config = json.loads(sys.argv[1])
normalized = module.normalize_config(config)
result = module.run_effgen(normalized)
print(json.dumps(result))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(mock_runner_script)
        mock_path = f.name

    try:
        config = {"task_type": "research", "user_goal": "mock goal", "preset": "research", "need_decomposition": False}
        result = subprocess.run(
            [sys.executable, mock_path, json.dumps(config)],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["mode_selected"] == "preset_research"
        assert output["output"] == "preset output"

        config_multi = {"task_type": "research", "user_goal": "mock goal", "need_decomposition": True}
        result = subprocess.run(
            [sys.executable, mock_path, json.dumps(config_multi)],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["mode_selected"] == "custom_multi_agent"
        assert output["output"] == "multi agent output"
    finally:
        Path(mock_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Unit tests for normalize_config with small HuggingFace model IDs
# ---------------------------------------------------------------------------

def test_normalize_config_preserves_unknown_keys():
    """normalize_config must carry through fields not listed in its defaults."""
    mod = _load_run_module()
    raw = {
        "task_type": "research",
        "user_goal": "Summarise recent ML papers",
        "model_name": "liquid-ai/LFM2-1B",
        "expected_output_schema": {"type": "object"},
        "constraints": "Keep it brief",
        "need_self_correction": True,
        "input_paths": ["/data/papers"],
        "working_directory": "/tmp/research",
    }
    normalized = mod.normalize_config(raw)

    # Standard defaults must be present
    assert normalized["task_type"] == "research"
    assert normalized["model_name"] == "liquid-ai/LFM2-1B"
    # Extra fields must be preserved (not silently dropped)
    assert normalized.get("expected_output_schema") == {"type": "object"}
    assert normalized.get("constraints") == "Keep it brief"
    assert normalized.get("need_self_correction") is True
    assert normalized.get("input_paths") == ["/data/papers"]
    assert normalized.get("working_directory") == "/tmp/research"


def test_normalize_config_lfm2_model():
    """LFM2-1B (<2 B) config is normalised correctly with defaults filled in."""
    mod = _load_run_module()
    raw = {
        "task_type": "general",
        "user_goal": "Explain transformer attention in one paragraph.",
        "model_name": "liquid-ai/LFM2-1B",
        "quantization": "4bit",
    }
    cfg = mod.normalize_config(raw)
    assert cfg["model_name"] == "liquid-ai/LFM2-1B"
    assert cfg["quantization"] == "4bit"
    assert cfg["task_type"] == "general"
    assert cfg["need_decomposition"] is False
    assert cfg["need_memory"] is False
    assert cfg["tools_needed"] == []
    assert cfg["timeout"] == 300


def test_select_mode_lfm2_research_preset():
    """Mode selection routes an LFM2 config with explicit preset to 'preset'."""
    mod = _load_run_module()
    cfg = mod.normalize_config({
        "task_type": "research",
        "user_goal": "Find recent papers on SSMs.",
        "model_name": "liquid-ai/LFM2-1B",
        "preset": "research",
    })
    assert mod.select_mode(cfg) == "preset"


def test_normalize_config_gemma_small_model():
    """Gemma-3 1B (<2 B) config is normalised correctly with defaults filled in."""
    mod = _load_run_module()
    raw = {
        "task_type": "qa",
        "user_goal": "Answer: what is the capital of France?",
        "model_name": "google/gemma-3-1b-it",
        "quantization": "4bit",
    }
    cfg = mod.normalize_config(raw)
    assert cfg["model_name"] == "google/gemma-3-1b-it"
    assert cfg["quantization"] == "4bit"
    assert cfg["task_type"] == "qa"
    assert cfg["preset"] is None


def test_select_mode_gemma_memory_augmented():
    """Gemma config with need_memory routes to memory_augmented_run."""
    mod = _load_run_module()
    cfg = mod.normalize_config({
        "task_type": "chat",
        "user_goal": "Continue our previous conversation.",
        "model_name": "google/gemma-3-1b-it",
        "need_memory": True,
    })
    assert mod.select_mode(cfg) == "memory_augmented_run"


def test_select_effgen_mode_script_lfm2():
    """select_effgen_mode.py routes LFM2 general config to a suitable preset."""
    config = {
        "task_type": "general",
        "user_goal": "Summarise this document.",
        "model_name": "liquid-ai/LFM2-1B",
    }
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input=json.dumps(config), capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"].startswith("preset_")
    assert output["preset"] is not None


def test_select_effgen_mode_script_gemma():
    """select_effgen_mode.py routes Gemma QA config to preset_minimal."""
    config = {
        "task_type": "qa",
        "user_goal": "What year was Python created?",
        "model_name": "google/gemma-3-1b-it",
    }
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input=json.dumps(config), capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "preset_minimal"
    assert output["preset"] == "minimal"


def test_select_effgen_mode_script_bad_json():
    """select_effgen_mode.py must report a parse_error when given invalid JSON."""
    result = subprocess.run(
        [sys.executable, _SELECT_SCRIPT],
        input="{ not valid json }", capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output.get("parse_error") is True
