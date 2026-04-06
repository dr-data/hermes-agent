# effGen Source Notes

This document summarizes the findings from reviewing effGen's official documentation and repository to inform the implementation of the Hermes skill.

## Sources Reviewed
- Homepage: `https://effgen.org`
- Documentation (Proxy): `https://effgen.org/docs/`
- GitHub Repository: `https://github.com/ctrl-gaurav/effGen`

## Findings

### Repository Structure & Implementation
- **Install**: Provided via PyPI (`pip install effgen` / `pip install effgen[vllm]`) and local script (`./install.sh`).
- **Imports**: The framework exposes `Agent`, `load_model`, and `AgentConfig`. Tools are in `effgen.tools.builtin`. Presets use `from effgen.presets import create_agent`.
- **Model Loading**: Supports loading HuggingFace models, with quantization options (e.g., `load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")`). Recommended default is Qwen2.5-3B-Instruct.
- **Presets**:
  - `math`
  - `research`
  - `coding`
  - `general`
  - `minimal`
- **Execution**: Run via `agent.run(prompt)` which returns a result object (e.g., `result.output`).
- **Memory**: Enabled via `enable_memory=True` in `AgentConfig`.
- **Multi-Agent**: Orchestration supported via A2A protocols.

### Discrepancies & API Uncertainties
- **Tool Names**: The homepage and repo indicate 14 tools, including `Calculator`, `PythonREPL`, `WebSearch`, `URLFetch`, `Wikipedia`, `CodeExecutor`, `FileOps`, `BashTool`, `JSONTool`, `WeatherTool`, etc. The exact import paths (e.g., `effgen.tools.builtin`) are derived from examples.
- **Execution Return Type**: Examples show `result.output` but the exact structure of `result` (metrics, errors, artifacts) is not fully documented in the surface scan. Wrapper script will assume it might be a dataclass or dict.
- **Graceful Failure**: Since effGen might not be installed, the Python integration script (`scripts/run_effgen_task.py`) must isolate imports and return structured JSON errors (`{"error": "effgen not installed"}`) rather than letting `ImportError` crash the process.

## Design Decisions for Hermes Integration
1. **Thin Adapter Pattern**: The `run_effgen_task.py` script will use a try-except block around `import effgen` to ensure it always outputs valid JSON.
2. **Preset Priority**: If a request matches a preset, `select_effgen_mode.py` will route to it.
3. **Structured Outputs**: All helper scripts will enforce strict JSON emission matching `templates/result-schema.json`.
4. **Mocking/Stubbing for Uninstalled State**: The script will honestly report missing dependencies.
