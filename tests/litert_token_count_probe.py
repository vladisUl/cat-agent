from __future__ import annotations

import os

import litert_lm


MODEL = os.getenv(
    "LITERT_AGENT_MODEL_PATH",
    "/storage/models/litertlm/gemma-4-E4B-it.litertlm",
)


def tokenized_prefill_count(engine: litert_lm.Engine, text: str, *, first_turn: bool) -> int:
    count = len(engine.tokenize(text))
    if first_turn and engine.bos_token_id is not None:
        count += 1
    return count


def run_benchmark_case() -> None:
    print("=== benchmark=True: compare official counters with Engine.tokenize() ===")
    with litert_lm.Engine(
        MODEL,
        backend=litert_lm.Backend.CPU(),
        enable_speculative_decoding=False,
        enable_benchmark=True,
    ) as engine:
        with engine.create_session(
            apply_prompt_template=False,
            max_output_tokens=64,
        ) as session:
            first = "Ответь одним словом: привет"
            predicted_first = tokenized_prefill_count(engine, first, first_turn=True)
            session.run_prefill([first])
            actual_first = session.get_benchmark_info().last_prefill_token_count

            response1 = session.run_decode()
            text1 = response1.texts[0] if response1.texts else ""
            actual_decode1 = session.get_benchmark_info().last_decode_token_count
            retokenized_decode1 = len(engine.tokenize(text1))

            second = "Теперь ответь одним словом: пока"
            predicted_second = tokenized_prefill_count(engine, second, first_turn=False)
            session.run_prefill([second])
            actual_second = session.get_benchmark_info().last_prefill_token_count

            response2 = session.run_decode()
            text2 = response2.texts[0] if response2.texts else ""
            actual_decode2 = session.get_benchmark_info().last_decode_token_count
            retokenized_decode2 = len(engine.tokenize(text2))

            print(
                f"prefill#1 official={actual_first} tokenize={predicted_first} "
                f"match={actual_first == predicted_first}"
            )
            print(
                f"decode#1  official={actual_decode1} retokenize={retokenized_decode1} "
                f"match={actual_decode1 == retokenized_decode1} text={text1!r}"
            )
            print(
                f"prefill#2 official={actual_second} tokenize={predicted_second} "
                f"match={actual_second == predicted_second}"
            )
            print(
                f"decode#2  official={actual_decode2} retokenize={retokenized_decode2} "
                f"match={actual_decode2 == retokenized_decode2} text={text2!r}"
            )


def run_no_benchmark_case() -> None:
    print("\n=== benchmark=False: verify BenchmarkInfo is unavailable ===")
    with litert_lm.Engine(
        MODEL,
        backend=litert_lm.Backend.CPU(),
        enable_speculative_decoding=False,
        enable_benchmark=False,
    ) as engine:
        with engine.create_session(
            apply_prompt_template=False,
            max_output_tokens=16,
        ) as session:
            session.run_prefill(["привет"])
            try:
                info = session.get_benchmark_info()
            except Exception as exc:
                print(f"get_benchmark_info: {type(exc).__name__}: {exc}")
            else:
                print(f"UNEXPECTED benchmark info: {info!r}")


if __name__ == "__main__":
    run_benchmark_case()
    run_no_benchmark_case()
