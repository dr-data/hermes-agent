#!/usr/bin/env python3
"""
run_effgen_task.py

Executes a task using effGen based on the provided configuration JSON.
It safely wraps the effGen library, providing structured JSON outputs even on failures.

Supports two backends:
  - ``effgen`` (default): loads a local HuggingFace model via the effGen library.
  - ``openrouter``: calls the OpenRouter API (OpenAI-compatible) using the
    ``OPENROUTER_API_KEY`` environment variable.  No local GPU or effGen
    installation is required for this path.

Input: JSON via stdin or CLI argument.
Output: Valid JSON matching result-schema.json.
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
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

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def normalize_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all expected fields exist with sensible defaults.

    Unknown keys from the caller (e.g. ``expected_output_schema``,
    ``constraints``, ``need_self_correction``, ``input_paths``,
    ``working_directory``) are preserved so that callers do not lose data.
    """
    defaults = {
        "task_type": "general",
        "user_goal": "No goal specified",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "quantization": "4bit",
        "preset": None,
        "tools_needed": [],
        "custom_tools": [],
        "need_decomposition": False,
        "need_memory": False,
        "timeout": 300,
        "save_artifacts": False,
        # OpenRouter-specific fields (optional)
        "api_backend": None,          # "openrouter" | None (default: effgen local)
        "openrouter_model": None,     # e.g. "liquid/lfm-2.5-1.2b-instruct"
    }

    normalized = dict(raw_config)
    for key, value in defaults.items():
        normalized.setdefault(key, value)

    return normalized


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
    """Return file paths written by effGen during the task.

    Currently returns an empty list because effGen does not yet expose a
    stable artifact-discovery API.  When ``save_artifacts`` is ``True`` the
    caller should inspect the working directory after the run to find any
    files written by the agent.
    """
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
    return build_error_result(
        "custom_multi_agent mode is not yet implemented for local effGen execution.",
        "Use a preset (e.g. preset_general) or the OpenRouter backend instead, "
        "or remove need_decomposition from the config.",
    )


def run_custom_tool_pipeline(config: Dict[str, Any], model: Any, goal: str) -> Dict[str, Any]:
    return build_error_result(
        "custom_tool_pipeline mode is not yet implemented for local effGen execution.",
        "Use preset_general with tools_needed, or the OpenRouter backend instead, "
        "or remove custom_tools from the config.",
    )


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


def _get_mode_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Determine the execution mode by delegating to select_effgen_mode.analyze_config().

    Imports the sibling script dynamically so there is no duplicate routing
    logic between the two scripts.  The returned dict has the same shape as
    ``analyze_config()`` output::

        {"mode_selected": "preset_research", "preset": "research", ...}
    """
    import importlib.util as _util
    _here = Path(__file__).resolve().parent
    _spec = _util.spec_from_file_location("select_effgen_mode", _here / "select_effgen_mode.py")
    _mod = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.analyze_config(config)


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

def run_via_openrouter(config: Dict[str, Any]) -> Dict[str, Any]:
    """Call the OpenRouter API (OpenAI-compatible) with no local GPU required.

    The API key is read from the ``OPENROUTER_API_KEY`` environment variable.
    The model is taken from ``config["openrouter_model"]``, defaulting to the
    value in ``config["model_name"]`` so callers can omit the extra field.

    Returns a result dict matching ``result-schema.json``.
    """
    import urllib.request
    import urllib.error

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return build_error_result(
            "OPENROUTER_API_KEY environment variable is not set.",
            "Set OPENROUTER_API_KEY to your OpenRouter API key before running this script.",
        )

    model = config.get("openrouter_model") or config["model_name"]
    goal = config["user_goal"]
    timeout_s = int(config.get("timeout", 60))

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": goal}],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_OPENROUTER_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dr-data/hermes-agent",
            "X-Title": "hermes-agent/effgen-orchestrator",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return build_error_result(
            f"OpenRouter API returned HTTP {exc.code}.",
            raw[:500],
        )
    except Exception as exc:
        return build_error_result("OpenRouter request failed.", str(exc))

    try:
        output_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return build_error_result("Unexpected OpenRouter response format.", str(body)[:500])

    usage = body.get("usage", {})
    return {
        "mode_selected": f"openrouter:{model}",
        "summary": f"Task completed via OpenRouter using model '{model}'.",
        "output": output_text,
        "artifacts": [],
        "errors": [],
        "metrics": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "next_action": "none",
        "suggested_skill_update": None,
    }


# ---------------------------------------------------------------------------

def run_effgen(config: Dict[str, Any]) -> Dict[str, Any]:
    """Core logic to initialize and run the task based on normalized config.

    When ``api_backend`` is ``"openrouter"``, the OpenRouter cloud API is used
    and no local effGen installation is required.  Otherwise the local effGen
    library is used.
    """
    if config.get("api_backend") == "openrouter":
        return run_via_openrouter(config)

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
    mode_info = _get_mode_info(config)
    mode = mode_info["mode_selected"]
    preset_name = mode_info.get("preset")

    try:
        if mode.startswith("preset_"):
            return run_preset_mode(preset_name or mode[len("preset_"):], model, goal)
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
