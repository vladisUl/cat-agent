from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
import shutil
import subprocess

import litert_lm
from litert_lm import _ffi


CANDIDATE_SYMBOLS = (
    "litert_lm_conversation_config_set_prefill_preface_on_init",
    "litert_lm_conversation_get_benchmark_info",
    "litert_lm_conversation_get_token_count",
    "litert_lm_conversation_render_message_to_string",
    "litert_lm_session_run_prefill",
    "litert_lm_session_run_decode",
    "litert_lm_session_get_benchmark_info",
    "litert_lm_engine_settings_enable_benchmark",
    "litert_lm_engine_settings_set_num_prefill_tokens",
    "litert_lm_engine_settings_set_num_decode_tokens",
    "litert_lm_engine_settings_set_use_ringbuffers_local_attention",
)


def _signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<unavailable>"


def _find_library(lib: object) -> Path | None:
    raw = getattr(lib, "_name", None)
    if isinstance(raw, str):
        p = Path(raw)
        if p.is_absolute() and p.is_file():
            return p

    package_dir = Path(litert_lm.__file__).resolve().parent
    preferred = package_dir / "liblitert-lm.so"
    if preferred.is_file():
        return preferred

    matches = sorted(package_dir.glob("*.so"))
    return matches[0] if matches else None


def _nm_exports(path: Path) -> list[str]:
    nm = shutil.which("nm")
    if not nm:
        return []
    cp = subprocess.run(
        [nm, "-D", "--defined-only", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        return []
    return cp.stdout.splitlines()


def main() -> int:
    try:
        version = importlib.metadata.version("litert-lm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    print(f"LITERT_LM_VERSION={version}")
    print(f"PACKAGE={Path(litert_lm.__file__).resolve()}")
    print(f"ENGINE_INIT={_signature(litert_lm.Engine)}")
    print(f"CREATE_CONVERSATION={_signature(litert_lm.Engine.create_conversation)}")
    print(f"CREATE_SESSION={_signature(litert_lm.Engine.create_session)}")

    conversation = getattr(litert_lm, "Conversation", None)
    session = getattr(litert_lm, "Session", None)

    if conversation is not None:
        for name in (
            "send_message",
            "render_message_to_string",
            "get_benchmark_info",
            "token_count",
        ):
            print(f"PY_CONVERSATION_{name.upper()}={hasattr(conversation, name)}")

    if session is not None:
        for name in (
            "run_prefill",
            "run_decode",
            "run_decode_async",
            "get_benchmark_info",
            "cancel_process",
        ):
            print(f"PY_SESSION_{name.upper()}={hasattr(session, name)}")

    lib = _ffi._get_lib()
    lib_path = _find_library(lib)
    print(f"LIB={lib_path if lib_path else getattr(lib, '_name', '<unknown>')}")

    for symbol in CANDIDATE_SYMBOLS:
        print(f"C_SYMBOL {symbol}={hasattr(lib, symbol)}")

    if lib_path is not None:
        exports = _nm_exports(lib_path)
        interesting = [
            line
            for line in exports
            if "litert_lm_" in line
            and any(
                key in line
                for key in (
                    "prefill",
                    "benchmark",
                    "conversation",
                    "session_run",
                    "token_count",
                    "render_message",
                    "ringbuffer",
                )
            )
        ]
        print("NM_INTERESTING_BEGIN")
        for line in interesting:
            print(line)
        print("NM_INTERESTING_END")

        demangler = shutil.which("c++filt")
        nm = shutil.which("nm")
        if nm and demangler:
            cp = subprocess.run(
                [nm, "-D", "--defined-only", str(lib_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            demangled = subprocess.run(
                [demangler],
                input=cp.stdout,
                capture_output=True,
                text=True,
                check=False,
            )
            cpp_hits = [
                line
                for line in demangled.stdout.splitlines()
                if "PrefillPreface" in line or "prefill_preface" in line
            ]
            print("CPP_PREFILL_SYMBOLS_BEGIN")
            for line in cpp_hits:
                print(line)
            print("CPP_PREFILL_SYMBOLS_END")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
