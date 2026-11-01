from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from litert_agent_tools.mqtt_cache import MqttCache


class CacheTests(unittest.TestCase):
    def test_reuses_connection_rejects_stale_and_clears_on_disconnect(self):
        now = [100.0]
        client = Mock()
        cache = MqttCache('host', 1883, ('sensor',), 30,
                          client_factory=lambda: client, clock=lambda: now[0])
        # Initial timeout starts exactly one subscription client.
        with self.assertRaises(RuntimeError):
            cache.read('sensor', 0)
        cache._on_connect(client, None, None, 0, None)
        client.subscribe.assert_called_once_with([('sensor', 0)])
        cache._on_message(client, None, SimpleNamespace(topic='sensor', payload=b'{"temperature":24}', retain=True))
        self.assertEqual(cache.read('sensor', 0), (b'{"temperature":24}', 0, True))
        now[0] += 2
        self.assertEqual(cache.read('sensor', 0)[1], 2)
        client.connect_async.assert_called_once()
        now[0] += 30
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            cache.read('sensor', 0)
        cache._on_disconnect(client, None, None, 1, None)
        self.assertEqual(cache._values, {})
        cache._on_connect(client, None, None, 0, None)
        with self.assertRaises(RuntimeError):
            cache.read('sensor', 0)
        cache.close()
        client.disconnect.assert_called_once()
        client.loop_stop.assert_called_once()

    def test_oversized_payload_invalidates_previous_value(self):
        client = Mock()
        cache = MqttCache('host', 1883, ('sensor',), 30, client_factory=lambda: client)
        cache._on_connect(client, None, None, 0, None)
        for payload in (b'{}', b'x' * 65537):
            cache._on_message(client, None, SimpleNamespace(topic='sensor', payload=payload, retain=False))
        self.assertNotIn('sensor', cache._values)


if __name__ == '__main__':
    unittest.main()
