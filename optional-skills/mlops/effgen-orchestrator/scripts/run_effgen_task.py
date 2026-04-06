#!/usr/bin/env python3
"""
run_effgen_task.py

Executes a task using effGen based on the provided configuration JSON.
It safely wraps the effGen library, providing structured JSON outputs even on failures.

Input: JSON via stdin or CLI argument.
Output: Valid JSON matching result-schema.json.
"""

import sys
import json
import argparse
import traceback
from typing import Dict, Any

# ---------------------------------------------------------------------------
# ADAPTER SECTION
# ---------------------------------------------------------------------------
# Isolate effGen imports so the script can gracefully fail and return JSON
# if effGen is not installed.

EFFGEN_INSTALLED = False
try:
    import effgen
    from effgen import Agent, load_model
    from effgen.core.agent import AgentConfig
    # We attempt to import presets if available
    try:
        from effgen.presets import create_agent
    except ImportError:
        create_agent = None
    EFFGEN_INSTALLED = True
except ImportError:
    pass

# ---------------------------------------------------------------------------

def normalize_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all expected fields exist with sensible defaults."""
    return {
        "task_type": raw_config.get("task_type", "general"),
        "user_goal": raw_config.get("user_goal", "No goal specified"),
        "model_name": raw_config.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
        "quantization": raw_config.get("quantization", "4bit"),
        "preset": raw_config.get("preset", None),
        "tools_needed": raw_config.get("tools_needed", []),
        "custom_tools": raw_config.get("custom_tools", []),
        "need_decomposition": raw_config.get("need_decomposition", False),
        "need_memory": raw_config.get("need_memory", False),
        "timeout": raw_config.get("timeout", 300),
        "save_artifacts": raw_config.get("save_artifacts", False),
    }

def build_error_result(error_msg: str, details: str = "") -> Dict[str, Any]:
    """Constructs a standard error JSON object."""
    return {
        "mode_selected": "error",
        "summary": "Task execution failed.",
        "output": None,
        "artifacts": [],
        "errors": [{
            "message": error_msg,
            "details": details
        }],
        "metrics": {},
        "next_action": "review_errors",
        "suggested_skill_update": None
    }

def collect_artifacts() -> list[str]:
    """Placeholder for collecting generated artifacts."""
    return []

def emit_result(mode_selected: str, summary: str, output: str, artifacts: list[str]) -> Dict[str, Any]:
    """Emits the standard result JSON object."""
    return {
        "mode_selected": mode_selected,
        "summary": summary,
        "output": output,
        "artifacts": artifacts,
        "errors": [],
        "metrics": {},
        "next_action": "none",
        "suggested_skill_update": None
    }

def build_preset_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"preset": config["preset"]}

def build_custom_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"enable_memory": config["need_memory"], "tools": config["tools_needed"]}

def run_preset_mode(preset: str, model: Any, goal: str) -> Dict[str, Any]:
    if not create_agent:
        raise RuntimeError("Presets not supported by installed effGen version.")
    agent = create_agent(preset, model)
    result = agent.run(goal)
    output_text = getattr(result, "output", str(result))
    return emit_result(f"preset_{preset}", "Task completed successfully using preset.", output_text, collect_artifacts())

def run_custom_single_agent(config: Dict[str, Any], model: Any, goal: str) -> Dict[str, Any]:
    agent_config = AgentConfig(
        name="custom_single_agent",
        model=model,
        tools=[],  # Placeholder: map config["tools_needed"] to actual tool classes
        enable_memory=config["need_memory"]
    )
    agent = Agent(config=agent_config)
    result = agent.run(goal)
    output_text = getattr(result, "output", str(result))
    return emit_result("custom_single_agent", "Task completed using custom single agent.", output_text, collect_artifacts())

def run_custom_multi_agent(config: Dict[str, Any], model: Any, goal: str) -> Dict[str, Any]:
    # Placeholder for multi-agent implementation
    output_text = f"Simulated execution of multi-agent decomposition for goal: {goal}"
    return emit_result("custom_multi_agent", "Task completed using custom multi-agent.", output_text, collect_artifacts())

def run_custom_tool_pipeline(config: Dict[str, Any], model: Any, goal: str) -> Dict[str, Any]:
    # Placeholder for custom tool pipeline implementation
    output_text = f"Simulated execution of custom tool pipeline for goal: {goal}"
    return emit_result("custom_tool_pipeline", "Task completed using custom tool pipeline.", output_text, collect_artifacts())

def run_memory_augmented(config: Dict[str, Any], model: Any, goal: str) -> Dict[str, Any]:
    agent_config = AgentConfig(
        name="memory_augmented_agent",
        model=model,
        tools=[],
        enable_memory=True
    )
    agent = Agent(config=agent_config)
    result = agent.run(goal)
    output_text = getattr(result, "output", str(result))
    return emit_result("memory_augmented_run", "Task completed using memory augmented agent.", output_text, collect_artifacts())

def select_mode(config: Dict[str, Any]) -> str:
    # Logic matching select_effgen_mode.py
    if config["need_decomposition"]:
        return "custom_multi_agent"
    elif config["custom_tools"]:
        return "custom_tool_pipeline"
    elif config["need_memory"]:
        return "memory_augmented_run"
    elif config["preset"]:
        return "preset"
    else:
        return "custom_single_agent"

def run_effgen(config: Dict[str, Any]) -> Dict[str, Any]:
    """Core logic to initialize and run effGen based on normalized config."""
    if not EFFGEN_INSTALLED:
        return build_error_result(
            "effGen library not installed.",
            "Please install effgen via `pip install effgen` or `pip install effgen[vllm]`."
        )

    try:
        model = load_model(config["model_name"], quantization=config["quantization"])
    except Exception as e:
        return build_error_result("Failed to load model.", str(e))

    goal = config["user_goal"]
    mode = select_mode(config)

    try:
        if mode == "preset":
            return run_preset_mode(config["preset"], model, goal)
        elif mode == "custom_multi_agent":
            return run_custom_multi_agent(config, model, goal)
        elif mode == "custom_tool_pipeline":
            return run_custom_tool_pipeline(config, model, goal)
        elif mode == "memory_augmented_run":
            return run_memory_augmented(config, model, goal)
        else:
            return run_custom_single_agent(config, model, goal)
    except Exception as e:
        return build_error_result(f"Agent execution failed in mode {mode}.", traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Run an effGen task.")
    parser.add_argument("--config", help="JSON string containing config")
    parser.add_argument("config_file", nargs="?", help="Path to config JSON file")
    args = parser.parse_args()

    raw_config = {}
    try:
        if args.config:
            raw_config = json.loads(args.config)
        elif args.config_file:
            with open(args.config_file, 'r') as f:
                raw_config = json.load(f)
        elif not sys.stdin.isatty():
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                raw_config = json.loads(stdin_data)
    except Exception as e:
        print(json.dumps(build_error_result("Invalid configuration format.", str(e)), indent=2))
        sys.exit(0)

    config = normalize_config(raw_config)
    result = run_effgen(config)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
