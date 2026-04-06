# effGen Capabilities

effGen is a local execution framework optimized for Small Language Models (SLMs) that provides a delegated execution layer for multi-agent workflows.

## Core Capabilities
- **SLM Optimization:** Uses vLLM for 5-10x faster inference, with automatic multi-GPU support and prompt optimization for smaller models.
- **Task Decomposition:** Analyzes complexity and automatically breaks down tasks into sub-agents.
- **Routing:** Complexity-aware routing logic to escalate tasks from simple single-agent to multi-agent.
- **Memory System:** Short-term, long-term, and vector memory integration for persistent multi-turn context.
- **Tool Integration:** Built-in and custom tools via MCP, A2A, and ACP protocols.

## Execution Modes & Presets
effGen provides one-line agent creation using presets. You should always prefer presets when possible.

### Presets
1. **`preset_minimal`**: Direct inference without tools. Best for quick QA or text transformation.
2. **`preset_general`**: All built-in tools included. Broad tasks requiring mixed capabilities.
3. **`preset_research`**: WebSearch, Wikipedia, and URLFetch. For gathering information and citations.
4. **`preset_coding`**: CodeExecutor, PythonREPL, FileOps, Bash. For local development and execution.
5. **`preset_math`**: Calculator, PythonREPL. Symbolic and numerical calculations.

### Custom Escalations
- **`custom_single_agent`**: For structured workflows needing a specific toolset not covered by presets.
- **`custom_multi_agent`**: For highly complex, decomposable tasks requiring specialized sub-agents.
- **`custom_tool_pipeline`**: When domain-specific plugins or non-standard tools are required.
- **`memory_augmented_run`**: For tasks requiring context across multiple interactions or vector search.

## Built-in Tools
- **Computation**: Calculator
- **Code**: CodeExecutor, PythonREPL
- **System**: FileOps, BashTool
- **Info/Web**: WebSearch, URLFetch, Wikipedia, Retrieval (RAG), AgenticSearch
- **Data**: JSONTool, TextProcessing
- **External**: WeatherTool, DateTimeTool

## Use Cases
- OCR cleanup and data extraction pipelines.
- Code generation, execution, and debugging.
- Comprehensive web research with fact-checking.
- Data analysis on local files.
- RAG knowledge base generation.

## Note on Compatibility
Capabilities, presets, and built-in tools may vary based on the locally installed effGen version. Helper scripts should degrade gracefully and handle missing tools or API mismatches by returning structured errors rather than crashing.
