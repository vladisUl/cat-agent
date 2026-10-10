"""Three real requests: cold base, independent request, dialogue continuation."""
import argparse
import json
from pathlib import Path
import time
from urllib import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9379/v1")
    parser.add_argument("--model", default="gemma4-12b")
    parser.add_argument("--system-file", help="UTF-8 file with the actual system prompt")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    system = (Path(args.system_file).read_text(encoding="utf-8-sig") if args.system_file else
              "Ты помощник. Отвечай коротко по-русски.")
    base = [{"role": "system", "content": system}]

    def send(label, messages):
        payload = {"model": args.model, "messages": messages, "stream": False,
                   "temperature": 0.0, "top_p": 1.0, "max_tokens": 64}
        started = time.monotonic()
        req = request.Request(args.url.rstrip("/") + "/chat/completions",
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
        usage = result["usage"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        print(f"{label}: seconds={time.monotonic() - started:.3f} "
              f"prompt={usage['prompt_tokens']} cached={cached} "
              f"evaluated={usage['prompt_tokens'] - cached} "
              f"decoded={usage['completion_tokens']}")
        message = result["choices"][0]["message"]
        print(message["content"])
        return message, cached

    messages = base + [{"role": "user", "content": "Привет!"}]
    send("1. First request", messages)
    reply, cached_base = send("2. Same request, fresh dialogue", messages)
    _, cached_dialogue = send("3. Continue dialogue", messages + [reply, {
        "role": "user", "content": "Сколько будет два плюс два?"
    }])
    if cached_base <= 0 or cached_dialogue <= 0:
        raise SystemExit("FAIL: expected cached tokens in requests 2 and 3; inspect server logs")
    print("PASS: base and subsequent context were reused. Check mode=resume for request 3; "
          "a reply truncated at 64 tokens requires replay from the base.")


if __name__ == "__main__":
    main()
