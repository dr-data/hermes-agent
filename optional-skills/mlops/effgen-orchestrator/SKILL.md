---
name: effgen-orchestrator
description: Use local effGen as a specialist SLM execution framework for presets, routing, decomposition, tools, memory-aware runs, and multi-agent workflows.
version: 1.0.0
platforms: [macos, linux]
meta:
  hermes:
    tags: [effgen, slm, agents, orchestration, automation, research, coding, ocr, rag, lfm2, gemma, qwen]
    category: mlops
    requires_toolsets: [terminal]
    tested_models:
      - liquid-ai/LFM2-1B
      - google/gemma-3-1b-it
      - Qwen/Qwen2.5-1.5B-Instruct
---

# effgen-orchestrator

## When to Use
Use this skill for tasks that benefit from local effGen execution, especially:
- Long or noisy context requiring prompt compression
- Decomposable complex workflows
- Automatic preset selection for common agent patterns
- Multi-step structured workflows
- Memory-aware repeated tasks
- Custom tool pipelines

## When Not to Use
Do not use this skill for:
- Tiny one-shot tasks
- Normal direct conversation
- Tasks Hermes can answer directly without delegation

## Capability Map
- quick direct no-tool inference -> `preset_minimal`
- broad general tool use -> `preset_general`
- web research and source gathering -> `preset_research`
- coding and local execution -> `preset_coding`
- numerical and symbolic tasks -> `preset_math`
- structured non-trivial workflow -> `custom_single_agent`
- decomposable complex workflow -> `custom_multi_agent`
- plugin or domain tool workflow -> `custom_tool_pipeline`
- repeated memory-sensitive workflow -> `memory_augmented_run`

## Inputs
- `task_type`: The broad category of the task (e.g., "coding", "research", "general").
- `user_goal`: The ultimate goal the user wants to achieve.
- `input_paths`: List of files or URLs needed as input.
- `working_directory`: Directory to execute the task in.
- `model_name`: HuggingFace model ID or local path.
- `quantization`: Quantization level (e.g., "4bit", "8bit").
- `preset`: Specific effGen preset to force (optional).
- `tools_needed`: List of built-in effGen tools to attach.
- `custom_tools`: List of custom tools or plugins.
- `need_decomposition`: Boolean, whether to use auto-decomposition.
- `need_memory`: Boolean, whether to enable memory.
- `need_self_correction`: Boolean, whether self-correction loops are required.
- `expected_output_schema`: JSON schema of the expected result.
- `constraints`: Specific task constraints.
- `timeout`: Maximum execution time in seconds.
- `save_artifacts`: Whether to save outputs to files.

## Routing Logic
1. First, decide if effGen is needed at all.
2. If yes, prefer an existing effGen preset (`preset_minimal`, `preset_general`, `preset_research`, `preset_coding`, `preset_math`).
3. Attach explicit built-in tools if the preset alone is insufficient.
4. Escalate to a custom single-agent only if presets and built-in tools do not cover the requirement.
5. Escalate to a multi-agent or memory-augmented mode only if explicitly required by task complexity.
6. Only propose narrower child skills after repeated successful use or explicit user request.

## Procedure
1. Inspect the request and decide whether to delegate to effGen.
2. Gather only the minimum required context.
3. Normalize the task configuration into the required schema.
4. Select the best effGen mode using `scripts/select_effgen_mode.py`.
5. Prefer existing presets and built-in tool combinations based on the reviewed docs and GitHub repo.
6. Run the local wrapper script `scripts/run_effgen_task.py`.
7. Capture stdout, stderr, artifacts, and structured JSON output.
8. Summarize the result back to the user.
9. Suggest narrower skill refinement only after repeated stable usage.

## Output Contract
Requires machine-readable structured output matching `templates/result-schema.json`, including:
- `mode_selected`
- `summary`
- `output`
- `artifacts`
- `errors`
- `metrics`
- `next_action`
- `suggested_skill_update`

## Pitfalls
- effGen not installed or missing dependencies.
- Unsupported preset specified.
- Missing or invalid model path.
- Missing custom tool dependency.
- Malformed configuration JSON.
- Oversized context exceeding SLM limits.
- Artifact path mismatch.
- Wrapper import error.
- Local execution failure due to resource constraints.
- Docs and repo API mismatch across versions.

## Verification
- Valid JSON output from wrapper scripts.
- Sensible mode selection based on inputs.
- Config normalization successful.
- Artifact collection successful.
- Structured failures returned instead of hard crashes.
- Correctness of preset-first routing.
- Alignment with actual effGen repo implementation.

## Examples
- OCR cleanup and correction
- Coding task using `preset_coding`
- Research task using `preset_research`
- Long-context extraction using prompt optimization
- Complex multi-step task using `custom_multi_agent`
- Memory-aware repeated workflow
- Data analysis workflow
- Weather or JSON API workflow
- RAG or document knowledge-base workflow

## Skill Evolution Rules
- Do not create narrower child skills immediately.
- Only propose child skills such as `effgen-ocr`, `effgen-research`, `effgen-coding`, `effgen-rag`, or `effgen-eval-correct` after repeated success or explicit user request.
- The main gateway skill remains the default entry point.
