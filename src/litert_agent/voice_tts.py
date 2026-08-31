from __future__ import annotations

from array import array
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any


_TTS_STOP = object()
TTS_TAIL_SILENCE_SECONDS = 0.50
POST_TTS_GUARD_SECONDS = 0.10
APLAY_DEVICE = os.environ.get("VLAD_APLAY_DEVICE", "").strip()

WAKE_BEEP_HIGH_FREQUENCY_HZ = 784.0
WAKE_BEEP_LOW_FREQUENCY_HZ = 523.25
WAKE_BEEP_SECONDS = 1.0
WAKE_BEEP_TONE_SECONDS = WAKE_BEEP_SECONDS / 2
WAKE_BEEP_SAMPLE_RATE = 24_000
WAKE_BEEP_AMPLITUDE = 0.12


def _wake_beep_pcm() -> bytes:
    tone_frames = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_TONE_SECONDS)
    peak = int(32767 * WAKE_BEEP_AMPLITUDE)
    samples = array("h")

    for frequency_hz in (
        WAKE_BEEP_HIGH_FREQUENCY_HZ,
        WAKE_BEEP_LOW_FREQUENCY_HZ,
    ):
        for frame in range(tone_frames):
            phase = 2.0 * math.pi * frequency_hz * frame / WAKE_BEEP_SAMPLE_RATE
            samples.append(round(peak * math.sin(phase)))

    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def play_wake_beep() -> None:
    command = ["aplay", "-q"]
    if APLAY_DEVICE:
        command.extend(["-D", APLAY_DEVICE])
    command.extend(
        [
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(WAKE_BEEP_SAMPLE_RATE),
            "-c",
            "1",
        ]
    )
    subprocess.run(command, input=_wake_beep_pcm(), check=True)


class StreamingTTSPlayer:
    """Stream Piper fragments through one persistent aplay process."""

    def __init__(self, piper_voice: Any, response_started_at: float) -> None:
        self._voice = piper_voice
        self._response_started_at = response_started_at
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="piper-stream",
            daemon=True,
        )
        self._error: BaseException | None = None
        self._first_audio_at: float | None = None
        self._playback_finished_at: float | None = None
        self._fragment_count = 0
        self._fragment_chars: list[int] = []

    def start(self) -> None:
        self._thread.start()

    def submit(self, text: str) -> None:
        fragment = " ".join(text.split())
        if fragment:
            self._queue.put(fragment)

    def finish(self) -> None:
        self._queue.put(_TTS_STOP)
        self._thread.join()

        if self._error is not None:
            raise RuntimeError(f"Ошибка потокового TTS: {self._error}") from self._error

        if self._first_audio_at is None:
            print("SKIP_TTS: пустой ответ")
            return

        print(f"TTS_FRAGMENTS: {self._fragment_count}")
        print("TTS_FRAGMENT_CHARS: " + ", ".join(map(str, self._fragment_chars)))
        print(f"VOICE_FIRST: {self._first_audio_at - self._response_started_at:.3f} с")
        if self._playback_finished_at is not None:
            print(
                "VOICE_TOTAL: "
                f"{self._playback_finished_at - self._response_started_at:.3f} с"
            )

    def _open_aplay(self, sample_rate: int, channels: int) -> subprocess.Popen[bytes]:
        command = ["aplay", "-q"]
        if APLAY_DEVICE:
            command.extend(["-D", APLAY_DEVICE])

        command.extend(
            [
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                str(channels),
            ]
        )

        process = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        if process.stdin is None:
            raise RuntimeError("Не удалось открыть stdin процесса aplay")
        return process

    def _worker(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        audio_format: tuple[int, int, int] | None = None

        try:
            while True:
                item = self._queue.get()
                if item is _TTS_STOP:
                    break

                assert isinstance(item, str)
                self._fragment_count += 1
                self._fragment_chars.append(len(item))

                for audio_chunk in self._voice.synthesize(item):
                    audio = audio_chunk.audio_int16_bytes
                    if not audio:
                        continue

                    current_format = (
                        int(audio_chunk.sample_rate),
                        int(audio_chunk.sample_width),
                        int(audio_chunk.sample_channels),
                    )
                    sample_rate, sample_width, channels = current_format

                    if sample_width != 2:
                        raise RuntimeError(
                            f"Piper вернул sample_width={sample_width}, ожидалось 2"
                        )

                    if process is None:
                        process = self._open_aplay(sample_rate, channels)
                        audio_format = current_format
                    elif current_format != audio_format:
                        raise RuntimeError(
                            "Piper изменил формат PCM внутри одного ответа: "
                            f"{audio_format} -> {current_format}"
                        )

                    if self._first_audio_at is None:
                        self._first_audio_at = time.monotonic()

                    assert process.stdin is not None
                    process.stdin.write(audio)

            if process is not None:
                assert process.stdin is not None
                assert audio_format is not None
                sample_rate, sample_width, channels = audio_format

                tail_frames = round(sample_rate * TTS_TAIL_SILENCE_SECONDS)
                tail_bytes = tail_frames * sample_width * channels
                process.stdin.write(b"\x00" * tail_bytes)
                process.stdin.close()

                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, process.args)

                self._playback_finished_at = time.monotonic()
                time.sleep(POST_TTS_GUARD_SECONDS)

        except BaseException as exc:
            self._error = exc
            if process is not None and process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except Exception:
                    pass

                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
