import json
import subprocess
import os

def test_select_effgen_mode():
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/select_effgen_mode.py"
    config_path = "optional-skills/mlops/effgen-orchestrator/templates/task-config.example.json"

    result = subprocess.run(["python3", script_path, config_path], capture_output=True, text=True)
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert output["mode_selected"] == "preset_research"
    assert output["preset"] == "research"

def test_run_effgen_task_missing_deps():
    # Because effGen is likely not installed, it should gracefully output JSON error
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/run_effgen_task.py"
    config_path = "optional-skills/mlops/effgen-orchestrator/templates/task-config.example.json"

    result = subprocess.run(["python3", script_path, config_path], capture_output=True, text=True)
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert output["mode_selected"] == "error"
    assert len(output["errors"]) > 0
    assert "effGen library not installed" in output["errors"][0]["message"]

def test_select_effgen_mode_multi_agent():
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/select_effgen_mode.py"
    config = {
        "task_type": "complex",
        "need_decomposition": True
    }

    result = subprocess.run(["python3", script_path], input=json.dumps(config), capture_output=True, text=True)
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "custom_multi_agent"
    assert output["need_decomposition"] is True

def test_select_effgen_mode_tool_pipeline():
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/select_effgen_mode.py"
    config = {
        "task_type": "general",
        "custom_tools": ["my_tool"]
    }

    result = subprocess.run(["python3", script_path], input=json.dumps(config), capture_output=True, text=True)
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "custom_tool_pipeline"
    assert output["need_custom_tools"] is True

def test_select_effgen_mode_memory():
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/select_effgen_mode.py"
    config = {
        "task_type": "chat",
        "need_memory": True
    }

    result = subprocess.run(["python3", script_path], input=json.dumps(config), capture_output=True, text=True)
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "memory_augmented_run"
    assert output["need_memory"] is True

def test_run_effgen_mock_installed(monkeypatch):
    script_path = "optional-skills/mlops/effgen-orchestrator/scripts/run_effgen_task.py"

    # We create a mock effgen runner that pretends effgen is installed
    mock_runner_script = """
import json
import sys

# Fake out the script by replacing run_effgen_task's installed flag
import importlib.util
spec = importlib.util.spec_from_file_location("run_effgen_task", "optional-skills/mlops/effgen-orchestrator/scripts/run_effgen_task.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules["run_effgen_task"] = module

# Mock effGen environment
module.EFFGEN_INSTALLED = True
class MockModel:
    pass
module.load_model = lambda name, quantization: MockModel()

# Mock different modes
module.create_agent = True
module.run_preset_mode = lambda p, m, g: module.emit_result("preset_" + p, "success", "preset output", [])
module.run_custom_multi_agent = lambda c, m, g: module.emit_result("custom_multi_agent", "success", "multi agent output", [])

config = json.loads(sys.argv[1])
normalized = module.normalize_config(config)
result = module.run_effgen(normalized)
print(json.dumps(result))
"""

    with open("mock_run.py", "w") as f:
        f.write(mock_runner_script)

    config = {
        "task_type": "research",
        "user_goal": "mock goal",
        "preset": "research",
        "need_decomposition": False
    }

    result = subprocess.run(["python3", "mock_run.py", json.dumps(config)], capture_output=True, text=True)
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "preset_research"
    assert output["output"] == "preset output"

    config_multi = {
        "task_type": "research",
        "user_goal": "mock goal",
        "need_decomposition": True
    }

    result = subprocess.run(["python3", "mock_run.py", json.dumps(config_multi)], capture_output=True, text=True)
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["mode_selected"] == "custom_multi_agent"
    assert output["output"] == "multi agent output"

    os.remove("mock_run.py")
