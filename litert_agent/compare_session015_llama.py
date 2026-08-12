from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import sys


# Canonical llama.cpp control result for exactly the same task/agent chain:
# manager -> agent -> tool -> second agent pass -> manager.
LLAMA_WALL_SECONDS = (4.561, 9.168, 14.421, 5.522)
PASS_NAMES = ("manager 1", "agent 1", "agent 2", "manager 2")

KV_RE = re.compile(
    r"litert015-session-kv (manager|agent) "
    r"resident=(\d+) new=(\d+) decode=(\d+) after=(\d+) "
    r"wall=([0-9.]+)s prefill=([0-9.]+)s generate=([0-9.]+)s"
)
CONTROL_RE = re.compile(r"^(CONTROL_[A-Z0-9_]+)=(.*)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PassResult:
    role: str
    resident: int
    new_tokens: int
    decode_tokens: int
    after: int
    wall: float
    prefill: float
    generate: float


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "litert_agent.session_015_temperature_shot"],
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        print("LiteRT control run FAILED", file=sys.stderr)
        if proc.stdout:
            print("===== STDOUT =====", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("===== STDERR =====", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    controls = dict(CONTROL_RE.findall(proc.stdout))
    passes = [
        PassResult(
            role=m.group(1),
            resident=int(m.group(2)),
            new_tokens=int(m.group(3)),
            decode_tokens=int(m.group(4)),
            after=int(m.group(5)),
            wall=float(m.group(6)),
            prefill=float(m.group(7)),
            generate=float(m.group(8)),
        )
        for m in KV_RE.finditer(proc.stderr)
    ]

    expected_roles = ["manager", "agent", "agent", "manager"]
    actual_roles = [p.role for p in passes]
    if actual_roles != expected_roles:
        print(
            "Unexpected LiteRT pass sequence: "
            f"expected={expected_roles} actual={actual_roles}",
            file=sys.stderr,
        )
        print(proc.stderr, file=sys.stderr)
        return 3

    result = controls.get("CONTROL_RESULT", "?")
    chain_total = _control_seconds(controls, "CONTROL_CHAIN_TOTAL")
    warm_manager = _control_seconds(controls, "CONTROL_WARM_MANAGER")
    warm_agent = _control_seconds(controls, "CONTROL_WARM_AGENT")
    engine_total = _control_seconds(controls, "CONTROL_ENGINE_INIT")

    print(f"RESULT: {result}")
    print()
    print(
        f"{'PASS':<12} {'LLAMA':>9} {'LITERT':>9} {'DELTA':>9} "
        f"{'NEW':>6} {'DECODE':>7} {'PREFILL':>9} {'GEN':>9}"
    )
    print("-" * 78)

    for name, llama, current in zip(PASS_NAMES, LLAMA_WALL_SECONDS, passes):
        delta = current.wall - llama
        print(
            f"{name:<12} {llama:>8.3f}s {current.wall:>8.3f}s "
            f"{delta:>+8.3f}s {current.new_tokens:>6d} "
            f"{current.decode_tokens:>7d} {current.prefill:>8.3f}s "
            f"{current.generate:>8.3f}s"
        )

    llama_wall = sum(LLAMA_WALL_SECONDS)
    litert_wall = sum(p.wall for p in passes)
    delta_wall = litert_wall - llama_wall
    delta_pct = 100.0 * delta_wall / llama_wall

    print("-" * 78)
    print(
        f"{'MODEL WALL':<12} {llama_wall:>8.3f}s {litert_wall:>8.3f}s "
        f"{delta_wall:>+8.3f}s ({delta_pct:+.1f}%)"
    )

    if chain_total is not None:
        host_overhead = chain_total - litert_wall
        print(f"LITERT CHAIN TOTAL : {chain_total:.3f}s")
        print(f"HOST/TOOL OVERHEAD : {host_overhead:.3f}s")

    print()
    if engine_total is not None:
        print(f"ENGINE INIT         : {engine_total:.3f}s")
    if warm_manager is not None and warm_agent is not None:
        print(
            f"PREFIX WARM         : manager {warm_manager:.3f}s + "
            f"agent {warm_agent:.3f}s (excluded from MODEL WALL)"
        )

    return 0


def _control_seconds(controls: dict[str, str], key: str) -> float | None:
    value = controls.get(key)
    if value is None:
        return None
    value = value.strip()
    if value.endswith("s"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
