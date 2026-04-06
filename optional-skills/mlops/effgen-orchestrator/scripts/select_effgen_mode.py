#!/usr/bin/env python3
"""
select_effgen_mode.py

Analyzes a task configuration JSON and determines the optimal effGen execution mode.
Prefers presets first, then escalates based on complexity.

Input: JSON via stdin or file path.
Output: JSON object containing mode selection and reasoning.
"""

import sys
import json
import argparse
from typing import Dict, Any

def analyze_config(config: Dict[str, Any]) -> Dict[str, Any]:
    task_type = config.get("task_type", "").lower()
    custom_tools = config.get("custom_tools", [])
    need_memory = config.get("need_memory", False)
    need_decomp = config.get("need_decomposition", False)

    # Defaults
    mode = "custom_single_agent"
    preset = None
    reason = "Fallback to custom single agent."

    if need_decomp:
        mode = "custom_multi_agent"
        reason = "Task explicitly requires decomposition into a multi-agent workflow."
    elif custom_tools:
        mode = "custom_tool_pipeline"
        reason = "Task requires custom domain tools not covered by presets."
    elif need_memory:
        mode = "memory_augmented_run"
        reason = "Task is memory-sensitive or part of a repeated workflow."
    else:
        # Try preset matching
        if task_type in ["math", "calculation"]:
            mode = "preset_math"
            preset = "math"
            reason = "Task matches math capabilities."
        elif task_type in ["code", "coding", "development"]:
            mode = "preset_coding"
            preset = "coding"
            reason = "Task matches coding and execution capabilities."
        elif task_type in ["research", "search", "web"]:
            mode = "preset_research"
            preset = "research"
            reason = "Task matches web research capabilities."
        elif task_type in ["qa", "chat", "direct", "minimal"]:
            mode = "preset_minimal"
            preset = "minimal"
            reason = "Simple task, no tools required."
        elif task_type in ["general", "mixed"]:
            mode = "preset_general"
            preset = "general"
            reason = "Broad task requiring a mix of built-in tools."
        else:
            # Check if tools are requested but no custom tools
            tools_needed = config.get("tools_needed", [])
            if tools_needed:
                mode = "preset_general"
                preset = "general"
                reason = "General task with tools needed; falling back to general preset."
            else:
                mode = "preset_minimal"
                preset = "minimal"
                reason = "Unknown task type with no tools; defaulting to minimal preset."

    return {
        "mode_selected": mode,
        "reason": reason,
        "preset": preset,
        "need_memory": need_memory,
        "need_decomposition": need_decomp,
        "need_custom_tools": bool(custom_tools)
    }

def main():
    parser = argparse.ArgumentParser(description="Select effGen mode based on config.")
    parser.add_argument("config_file", nargs="?", help="Path to config JSON (optional, defaults to stdin)")
    args = parser.parse_args()

    config_data = {}
    try:
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
        else:
            if not sys.stdin.isatty():
                stdin_data = sys.stdin.read().strip()
                if stdin_data:
                    config_data = json.loads(stdin_data)
    except Exception as e:
        # Never crash on parsing errors, return fallback
        pass

    result = analyze_config(config_data)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
