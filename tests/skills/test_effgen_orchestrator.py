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
