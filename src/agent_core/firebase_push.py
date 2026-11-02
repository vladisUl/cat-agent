"""Shared Firebase transport for native and OpenAI CORE notification queues."""
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import threading

LOGGER = logging.getLogger(__name__)
DEFAULT_FIREBASE_DIR = Path("/opt/firebase")
FIREBASE_CREDENTIALS_FILE = Path(
    os.environ.get(
        "CAT_AGENT_FIREBASE_CREDENTIALS",
        str(DEFAULT_FIREBASE_DIR / "zigbee.json"),
    )
)
FIREBASE_TOKENS_FILE = Path(
    os.environ.get(
        "CAT_AGENT_FIREBASE_TOKENS",
        str(DEFAULT_FIREBASE_DIR / "tokens.txt"),
    )
)
FIREBASE_TITLE = os.environ.get("CAT_AGENT_FIREBASE_TITLE", "Гена").strip() or "Гена"
FIREBASE_CHANNEL_ID = os.environ.get("CAT_AGENT_FIREBASE_CHANNEL_ID", "").strip()
FIREBASE_TTL_SECONDS = int(os.environ.get("CAT_AGENT_FIREBASE_TTL", "3600"))


def read_firebase_tokens(path: Path) -> list[str]:
    """Read JSONL {id, token} records, keeping the last token for each id."""
    tokens_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"invalid token record in {path}:{line_number}")
            token_id = str(item.get("id", "")).strip()
            token = str(item.get("token", "")).strip()
            if not token_id or not token:
                raise ValueError(
                    f"token record requires non-empty id and token in {path}:{line_number}"
                )
            tokens_by_id[token_id] = token
    return list(tokens_by_id.values())


class FirebasePushSender:
    def __init__(self):
        self._firebase_lock = threading.Lock()

    def send(self, text: str) -> None:
        with self._firebase_lock:
            if not FIREBASE_CREDENTIALS_FILE.is_file():
                raise FileNotFoundError(
                    f"Firebase credentials not found: {FIREBASE_CREDENTIALS_FILE}"
                )
            if not FIREBASE_TOKENS_FILE.is_file():
                raise FileNotFoundError(
                    f"Firebase token file not found: {FIREBASE_TOKENS_FILE}"
                )

            try:
                import firebase_admin
                from firebase_admin import credentials, messaging
            except ImportError as exc:
                raise RuntimeError(
                    "firebase-admin is not installed in the cat-agent Python environment"
                ) from exc

            try:
                firebase_admin.get_app()
            except ValueError:
                firebase_admin.initialize_app(
                    credentials.Certificate(str(FIREBASE_CREDENTIALS_FILE))
                )

            tokens = read_firebase_tokens(FIREBASE_TOKENS_FILE)
            if not tokens:
                raise RuntimeError(
                    f"Firebase token file is empty: {FIREBASE_TOKENS_FILE}"
                )

            android_notification = None
            if FIREBASE_CHANNEL_ID:
                android_notification = messaging.AndroidNotification(
                    channel_id=FIREBASE_CHANNEL_ID
                )

            android = messaging.AndroidConfig(
                priority="high",
                ttl=timedelta(seconds=FIREBASE_TTL_SECONDS),
                notification=android_notification,
            )
            notification = messaging.Notification(
                title=FIREBASE_TITLE,
                body=text,
            )

            success = 0
            failure = 0
            for offset in range(0, len(tokens), 500):
                batch = tokens[offset : offset + 500]
                message = messaging.MulticastMessage(
                    tokens=batch,
                    notification=notification,
                    android=android,
                )
                response = messaging.send_each_for_multicast(message)
                success += response.success_count
                failure += response.failure_count
                if response.failure_count:
                    for index, result in enumerate(response.responses):
                        if not result.success:
                            LOGGER.warning(
                                "Firebase token delivery failed index=%d error=%s",
                                offset + index,
                                result.exception,
                            )

            LOGGER.info(
                "Firebase notification sent success=%d failure=%d",
                success,
                failure,
            )
            if failure:
                raise RuntimeError(f"Firebase delivery failed for {failure} tokens")

