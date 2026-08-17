# cat-agent

Experimental local LLM agent runtime.

The project explores manager/agent orchestration, prepared skills, local inference backends, persistent model state and tool execution on edge hardware.

Python handles runtime mechanics, transport, tools and state. Models operate inside supplied prompts and skills and decide how to carry out delegated work.

Source layout:
- `src/orchestration/` — shared manager/agent orchestration core used by all backends.
- `src/llama_agent/` — llama.cpp/OpenAI-compatible frontend.
- `src/litert_agent/` — LiteRT-LM frontend/runtime.

Launchers:
- `start_llama_server.sh` — start the llama.cpp server.
- `start_llama_agent.sh` — start cat-agent through the llama.cpp/OpenAI-compatible backend.
- `start_litert_agent.sh` — start cat-agent through LiteRT-LM.

The repository contains a working research baseline and will continue to evolve.

**To be continued.**
