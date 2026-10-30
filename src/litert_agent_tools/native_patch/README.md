# Enabling a genuinely prefilled Conversation BASE

## Why this needs a native rebuild

Upstream LiteRT-LM **0.18.0** already implements the correct C++ path:
"ConversationConfig::Builder().SetPrefillPrefaceOnInit(true)". That path
prefills the rendered system/tools preface without generating a synthetic
conversation turn. Conversation::Clone() then clones the actual native session
KV. The installed Python/C API for 0.18.0 does not expose the builder option.

There is NO safe ctypes-only invocation that changes this boolean on a stock
0.18.0 shared library. Avoid guessing offsets of internal C++ structures and
avoid fake initialization requests that pollute dialogue history.

## Minimal source extension

The source patch script adds one C config setter and wires the existing
C++ builder flag through the 0.18.0 C/Python bindings:

- c/conversation.h: declare setter;
- c/conversation_internal.h: add bool to private config;
- c/conversation.cc: implement setter and call SetPrefillPrefaceOnInit;
- python/litert_lm/_ffi.py: declare ctypes signature;
- python/litert_lm/engine.py: add optional prefill_preface_on_init=False.

The default remains FALSE: other LiteRT-LM users are unaffected. No change to
the old src/litert_agent Session backend is necessary.

### Applying

Use a SEPARATE Google LiteRT-LM source checkout at **v0.18.0**:

    cd /opt/cat-agent
    python3 src/litert_agent_tools/native_patch/enable_prefill.py --check /path/to/LiteRT-LM
    python3 src/litert_agent_tools/native_patch/enable_prefill.py /path/to/LiteRT-LM

This edits that specified LiteRT-LM source checkout only. It does NOT modify
the installed /opt/litert-lm-venv package, and installing a plain upstream
0.18 wheel again will remove the capability. Rebuild and reinstall a matched
native library + Python binding into the test environment using the LiteRT-LM
project's build process. A Python-only edit will NOT work: the new C symbol
must actually exist in the rebuilt shared library.

The script is idempotent and rejects unknown source layouts, but build/test
the native package separately. No rebuilt binary is included here.

### Verification after rebuilding

    cd /opt/cat-agent
    PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
      -m litert_agent_tools \
      --model /storage/models/litert/gemma-4-E4B-it.litertlm \
      --system-file /opt/prompt/prompts_tools/manager.md \
      --require-prefilled-base

Expected log sequence when the native patch is truly active:

- BASE prefill_supported=True kv_tokens=positive number
- BASE-CLONE kv_tokens=same positive number expected=same
- TURN kv_before=same base count kv_after=larger count

last_prefill_tokens reports the LAST native prefill pass, which can be
associated with a tool result during automatic tool calling; it is NOT the
total of the entire turn and should not be compared to the complete BASE size
without accounting for the tool-call loop.

Use /new repeatedly: the BASE-CLONE count must remain identical. This verifies
native KV inheritance; evaluate speed separately using timing data from the
same prompt/model/backend. If the flag is absent the CLI reports BASE
prefilled=False rather than making a false cache-reuse claim.
