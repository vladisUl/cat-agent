from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any
import wave

from .core_server import DEFAULT_CORE_SOCKET
from .voice_stream import ManagerVoiceStream, TextFragmenter
from .voice_tts import StreamingTTSPlayer, play_wake_beep


VOSK_MODEL_PATH = Path(
    os.environ.get("CAT_AGENT_VOSK_MODEL", "/opt/vosk/vosk-model")
)
PIPER_MODEL_PATH = Path(
    os.environ.get(
        "CAT_AGENT_PIPER_MODEL",
        "/opt/piper/voices/ru_RU-irina-medium.onnx",
    )
)
CORE_SOCKET = Path(
    os.environ.get("CAT_AGENT_CORE_SOCKET", str(DEFAULT_CORE_SOCKET))
)
COMMAND_WAV_PATH = Path(
    os.environ.get("CAT_AGENT_COMMAND_WAV", "/tmp/cat_agent_voice_command.wav")
)

GIGAAM_MODEL_NAME = os.environ.get("CAT_AGENT_GIGAAM_MODEL", "e2e_ctc")
AUDIO_DEVICE = os.environ.get("CAT_AGENT_AUDIO_DEVICE", "plughw:0,2")
WAKE_WORDS = {"гена"}

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
PERIOD_FRAMES = 800
ALSA_PERIODS = 8
NO_COMMAND_TIMEOUT_SECONDS = 7.0
MAX_COMMAND_SECONDS = 20.0


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} не найден: {path}")


def _json_text(raw_json: str, field: str) -> str:
    try:
        value = json.loads(raw_json).get(field, "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _make_wake_recognizer(vosk_module: Any, model: Any) -> Any:
    grammar = json.dumps([*sorted(WAKE_WORDS), "[unk]"], ensure_ascii=False)
    return vosk_module.KaldiRecognizer(model, SAMPLE_RATE, grammar)


def _make_command_recognizer(vosk_module: Any, model: Any) -> Any:
    return vosk_module.KaldiRecognizer(model, SAMPLE_RATE)


def _open_microphone(alsaaudio_module: Any) -> Any:
    return alsaaudio_module.PCM(
        type=alsaaudio_module.PCM_CAPTURE,
        mode=alsaaudio_module.PCM_NORMAL,
        device=AUDIO_DEVICE,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        format=alsaaudio_module.PCM_FORMAT_S16_LE,
        periodsize=PERIOD_FRAMES,
        periods=ALSA_PERIODS,
    )


def _open_command_wav(path: Path) -> wave.Wave_write:
    wav = wave.open(str(path), "wb")
    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH_BYTES)
    wav.setframerate(SAMPLE_RATE)
    return wav


def _gigaam_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()
    return str(result).strip()


def _transcribe_command(gigaam_model: Any, wav_path: Path) -> str:
    started = time.monotonic()
    result = gigaam_model.transcribe(str(wav_path))
    text = _gigaam_text(result)
    elapsed = time.monotonic() - started
    print(f"GIGAAM_TIME: {elapsed:.3f} с")
    print(f"USER: {text or '[пусто]'}")
    return text


def _send_json(sock: socket.socket, payload: dict[str, object]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    sock.sendall(data)


def _run_core_voice_turn(user_text: str, piper_voice: Any) -> str:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    reader = None
    started = time.monotonic()
    stream = ManagerVoiceStream()
    fragmenter = TextFragmenter()
    speaker = StreamingTTSPlayer(piper_voice, started)
    speaker_started = False
    streamed_visible = False
    printed_stream = False

    def submit_speech(text: str) -> None:
        nonlocal speaker_started
        if not text:
            return
        for fragment in fragmenter.feed(text):
            if not speaker_started:
                speaker.start()
                speaker_started = True
            speaker.submit(fragment)

    try:
        sock.connect(str(CORE_SOCKET))
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        _send_json(
            sock,
            {
                "type": "voice",
                "client": "voice",
                "text": user_text,
            },
        )
        print(f"CORE_VOICE_SENT: {user_text}")

        for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"CORE_BAD_JSON: {line!r}")
                continue
            if not isinstance(item, dict):
                continue

            kind = str(item.get("type", ""))

            if kind == "voice_accepted":
                print(f"CORE_VOICE_ACCEPTED: priority={item.get('priority')}")
                continue

            if kind == "busy":
                print(f"CORE_BUSY: {item.get('text', 'Гена занят')}")
                return ""

            if kind == "model_event" and item.get("label") == "manager":
                event = str(item.get("event", ""))
                payload = str(item.get("payload", ""))
                if event == "decode_start":
                    stream.decode_start()
                    continue
                if event != "chunk":
                    continue

                visible = stream.feed_chunk(payload)
                if not visible:
                    continue

                streamed_visible = True
                if not printed_stream:
                    print("GENA: ", end="", flush=True)
                    printed_stream = True
                print(visible, end="", flush=True)
                submit_speech(visible)
                continue

            if kind == "reply":
                final_text = str(item.get("text", "")).strip()
                if printed_stream:
                    print()
                elif final_text:
                    print(f"GENA: {final_text}")

                if final_text and not streamed_visible:
                    submit_speech(final_text)

                for fragment in fragmenter.finish():
                    if not speaker_started:
                        speaker.start()
                        speaker_started = True
                    speaker.submit(fragment)

                return final_text

            if kind == "error":
                raise RuntimeError(str(item.get("error", "CORE error")))

        raise RuntimeError("CORE закрыл voice-соединение до ответа")

    finally:
        if speaker_started:
            speaker.finish()
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass


def _wait_message() -> None:
    words = " или ".join(f"«{word}»" for word in sorted(WAKE_WORDS))
    print(f"WAIT_WAKE: скажи {words}, сделай короткую паузу, затем команду")


def _run_loop(
    alsaaudio_module: Any,
    vosk_module: Any,
    vosk_model: Any,
    gigaam_model: Any,
    piper_voice: Any,
) -> None:
    wake_recognizer = _make_wake_recognizer(vosk_module, vosk_model)
    command_recognizer: Any | None = None

    pcm = _open_microphone(alsaaudio_module)
    print(f"ALSA_READY: {AUDIO_DEVICE}, {SAMPLE_RATE} Гц, mono S16_LE")

    state = "wait_wake"
    command_wav: wave.Wave_write | None = None
    command_started_at = 0.0
    speech_started = False
    command_bytes = 0

    _wait_message()

    try:
        while True:
            try:
                frames, data = pcm.read()
            except alsaaudio_module.ALSAAudioError as exc:
                print(f"ALSA_ERROR: {exc}")
                pcm.close()
                time.sleep(0.2)
                pcm = _open_microphone(alsaaudio_module)
                print("ALSA_REOPENED")
                continue

            if frames == -errno.EPIPE:
                print("ALSA_OVERRUN")
                continue
            if frames < 0:
                print(f"ALSA_READ_ERROR: frames={frames}")
                continue
            if frames == 0 or not data:
                continue

            if state == "wait_wake":
                endpoint = wake_recognizer.AcceptWaveform(data)
                if not endpoint:
                    continue

                wake_text = _json_text(wake_recognizer.Result(), "text")
                if wake_text not in WAKE_WORDS:
                    if wake_text:
                        print(f"WAKE_REJECT: {wake_text}")
                    continue

                print(f"WAKE_FINAL: {wake_text}")

                # Do not let the acknowledgement tone leak into command capture.
                # Capture is reopened after the synchronous beep, which also clears
                # any samples accumulated while the speaker was active.
                pcm.close()
                try:
                    play_wake_beep()
                except Exception as exc:
                    print(f"WAKE_BEEP_ERROR: {type(exc).__name__}: {exc}")
                finally:
                    pcm = _open_microphone(alsaaudio_module)

                command_recognizer = _make_command_recognizer(vosk_module, vosk_model)
                command_wav = _open_command_wav(COMMAND_WAV_PATH)
                command_started_at = time.monotonic()
                speech_started = False
                command_bytes = 0
                state = "wait_command"
                print("COMMAND_WAIT: говори")
                continue

            assert command_recognizer is not None
            assert command_wav is not None

            command_wav.writeframesraw(data)
            command_bytes += len(data)

            endpoint = command_recognizer.AcceptWaveform(data)
            command_ready = False
            cancelled = False

            if endpoint:
                full_text = _json_text(command_recognizer.Result(), "text")
                if full_text:
                    speech_started = True
                    command_ready = True
                    print(f"COMMAND_FINAL_VOSK: {full_text}")
                else:
                    print("SILENCE_ENDPOINT")
            else:
                partial = _json_text(command_recognizer.PartialResult(), "partial")
                if partial:
                    speech_started = True

            elapsed = time.monotonic() - command_started_at
            if not speech_started and elapsed >= NO_COMMAND_TIMEOUT_SECONDS:
                print(f"NO_COMMAND: {elapsed:.3f} с")
                cancelled = True
            elif elapsed >= MAX_COMMAND_SECONDS:
                print(f"COMMAND_TIMEOUT: {elapsed:.3f} с")
                cancelled = True

            if not command_ready and not cancelled:
                continue

            command_wav.close()
            command_wav = None
            duration = command_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES)

            if cancelled:
                COMMAND_WAV_PATH.unlink(missing_ok=True)
                print("CANCELLED")
            else:
                print(f"AUDIO_READY: {COMMAND_WAV_PATH} duration={duration:.3f} с")

                # Keep the proven behaviour: close capture during heavy STT/agent/TTS
                # processing and reopen ALSA with an empty buffer afterwards.
                pcm.close()
                try:
                    user_text = _transcribe_command(gigaam_model, COMMAND_WAV_PATH)
                    if user_text:
                        _run_core_voice_turn(user_text, piper_voice)
                    else:
                        print("SKIP_CORE: GigaAM вернула пустой текст")
                except Exception as exc:
                    print(f"PROCESSING_ERROR: {type(exc).__name__}: {exc}")

                pcm = _open_microphone(alsaaudio_module)
                print("ALSA_RESUMED")

            wake_recognizer = _make_wake_recognizer(vosk_module, vosk_model)
            command_recognizer = None
            state = "wait_wake"
            _wait_message()

    finally:
        if command_wav is not None:
            command_wav.close()
        pcm.close()


def main() -> None:
    import alsaaudio
    import gigaam
    import vosk
    from piper import PiperVoice

    _require_path(VOSK_MODEL_PATH, "Модель Vosk")
    _require_path(PIPER_MODEL_PATH, "Модель Piper")
    if shutil.which("aplay") is None:
        raise FileNotFoundError("Команда aplay не найдена")

    vosk.SetLogLevel(-1)

    started = time.monotonic()
    print("INIT_VOSK...")
    vosk_model = vosk.Model(str(VOSK_MODEL_PATH))
    print(f"VOSK_READY: {time.monotonic() - started:.3f} с")

    started = time.monotonic()
    print(f"INIT_GIGAAM: {GIGAAM_MODEL_NAME}...")
    gigaam_model = gigaam.load_model(GIGAAM_MODEL_NAME)
    print(f"GIGAAM_READY: {time.monotonic() - started:.3f} с")

    started = time.monotonic()
    print("INIT_PIPER: ru_RU-irina-medium...")
    piper_voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    print(f"PIPER_READY: {time.monotonic() - started:.3f} с")

    print(f"CORE_SOCKET: {CORE_SOCKET}")
    _run_loop(
        alsaaudio,
        vosk,
        vosk_model,
        gigaam_model,
        piper_voice,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSTOP")
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        raise
