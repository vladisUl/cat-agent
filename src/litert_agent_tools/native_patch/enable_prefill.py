"""Apply the minimal LiteRT-LM v0.18.0 source patch for warmed Conversation BASE.

This patches GOOGLE'S LiteRT-LM source tree, NOT the cat-agent checkout or
the installed wheel. Rebuild/reinstall LiteRT-LM after applying.

Usage:
    python3 enable_prefill.py --check /path/to/LiteRT-LM
    python3 enable_prefill.py /path/to/LiteRT-LM

Only exact, known 0.18.0 snippets are accepted. Existing edits are not
overwritten silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CHANGES: dict[str, list[tuple[str, str]]] = {
    "c/conversation.h": [(
        """// Sets the initial messages for this conversation config.
""",
        """// Preload the preface (system message and tools) into the KV cache on
// Conversation creation, before the first user message.
// Added as a local extension to 0.18.0; not part of the stock C ABI.
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_set_prefill_preface_on_init(
    LiteRtLmConversationConfig* config, bool enable);

// Sets the initial messages for this conversation config.
""",
    )],
    "c/conversation_internal.h": [(
        """  bool enable_constrained_decoding = false;
""",
        """  bool enable_constrained_decoding = false;
  bool prefill_preface_on_init = false;
""",
    )],
    "c/conversation.cc": [
        (
            """void litert_lm_conversation_config_set_messages(
""",
            """void litert_lm_conversation_config_set_prefill_preface_on_init(
    LiteRtLmConversationConfig* config, bool enable) {
  if (config) {
    config->prefill_preface_on_init = enable;
  }
}

void litert_lm_conversation_config_set_messages(
""",
        ),
        (
            """    builder.SetPreface(json_preface);
    builder.SetEnableConstrainedDecoding(c_config->enable_constrained_decoding);
""",
            """    builder.SetPreface(json_preface);
    builder.SetPrefillPrefaceOnInit(c_config->prefill_preface_on_init);
    builder.SetEnableConstrainedDecoding(c_config->enable_constrained_decoding);
""",
        ),
    ],
    "python/litert_lm/_ffi.py": [(
        """  lib.litert_lm_conversation_config_set_messages.argtypes = [
""",
        """  lib.litert_lm_conversation_config_set_prefill_preface_on_init.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_conversation_config_set_messages.argtypes = [
""",
    )],
    "python/litert_lm/engine.py": [
        (
            """      enable_speculative_decoding: bool | None = None,
  ) -> Conversation:
""",
            """      enable_speculative_decoding: bool | None = None,
      prefill_preface_on_init: bool = False,
  ) -> Conversation:
""",
        ),
        (
            """      conv_ptr = self._lib.litert_lm_conversation_create(
          self._engine_ptr, conv_config
      )
""",
            """      if prefill_preface_on_init:
        self._lib.litert_lm_conversation_config_set_prefill_preface_on_init(
            conv_config, True
        )
      conv_ptr = self._lib.litert_lm_conversation_create(
          self._engine_ptr, conv_config
      )
""",
        ),
    ],
}


def change_source(root: Path, *, check: bool) -> list[str]:
    # Check every file and planned replacement before mutating any file.
    updated: dict[Path, str] = {}
    summary: list[str] = []
    for relative, replacements in CHANGES.items():
        path = root / relative
        old_text = path.read_text(encoding="utf-8")
        new_text = old_text
        changed = False
        for before, after in replacements:
            if after in new_text:
                continue
            if new_text.count(before) != 1:
                raise RuntimeError(
                    f"Unsupported source layout {relative}: expected exactly "
                    f"one matching 0.18.0 insertion point"
                )
            new_text = new_text.replace(before, after, 1)
            changed = True
        if changed:
            updated[path] = new_text
        summary.append(f'{"PATCH" if changed else "OK"} {relative}')
    if not check:
        for path, new_text in updated.items():
            path.write_text(new_text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate but do not write")
    parser.add_argument("source_dir", type=Path)
    options = parser.parse_args()
    for line in change_source(options.source_dir, check=options.check):
        print(line)
    if options.check:
        print("Check passed. No files changed.")


if __name__ == "__main__":
    main()
