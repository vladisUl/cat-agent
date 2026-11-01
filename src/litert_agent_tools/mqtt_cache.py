"""Optional persistent subscriptions; never serve a disconnected/stale cache."""
from __future__ import annotations

import threading
import time


class MqttCache:
    def __init__(self, host, port, topics, max_age, *, client_factory=None, clock=time.monotonic):
        self.host, self.port = host, port
        self.topics = tuple(dict.fromkeys(topics))
        self.max_age = max_age
        self.clock = clock
        self._factory = client_factory
        self._cv = threading.Condition()
        self._client = None
        self._connected = False
        self._closed = False
        self._values = {}

    def _start(self):
        # Called with _cv held. Paho callbacks run on its network thread.
        if self._client is not None:
            return
        if self._factory is None:
            try:
                import paho.mqtt.client as mqtt
            except ImportError as exc:
                raise RuntimeError('MQTT cache requires cat-agent[mqtt-cache] (paho-mqtt 2.x)') from exc
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        else:
            client = self._factory()
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client = client
        try:
            client.connect_async(self.host, self.port, keepalive=15)
            client.loop_start()
        except Exception:
            self._client = None
            client.disconnect()
            raise

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        with self._cv:
            if self._closed:
                return
            self._values.clear()
            self._connected = reason_code == 0
            if self._connected:
                client.subscribe([(topic, 0) for topic in self.topics])
            self._cv.notify_all()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        with self._cv:
            self._connected = False
            self._values.clear()
            self._cv.notify_all()

    def _on_message(self, client, userdata, message):
        with self._cv:
            if self._closed or not self._connected or message.topic not in self.topics:
                return
            if len(message.payload) > 65536:
                self._values.pop(message.topic, None)
            else:
                self._values[message.topic] = (
                    bytes(message.payload), self.clock(), bool(message.retain),
                )
            self._cv.notify_all()

    def read(self, topic, timeout):
        deadline = self.clock() + timeout
        with self._cv:
            if self._closed:
                raise RuntimeError('MQTT cache is closed')
            self._start()
            while not self._closed:
                item = self._values.get(topic) if self._connected else None
                if item is not None:
                    payload, received_at, retained = item
                    age = self.clock() - received_at
                    if age <= self.max_age:
                        return payload, age, retained
                    # No silent fallback to stale data or a new one-shot process.
                    raise RuntimeError('MQTT cached message expired')
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise RuntimeError('MQTT cache has no connected, fresh message')
                self._cv.wait(remaining)
            raise RuntimeError('MQTT cache is closed')

    def close(self):
        with self._cv:
            self._closed = True
            self._connected = False
            self._values.clear()
            client, self._client = self._client, None
            self._cv.notify_all()
        if client is not None:
            client.disconnect()
            client.loop_stop()


class MqttTools(list):
    def __init__(self, tools, caches=()):
        super().__init__(tools)
        self.caches = tuple(caches)

    def close(self):
        for cache in self.caches:
            cache.close()
