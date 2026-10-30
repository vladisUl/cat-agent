"""Stage-2 native manager IPC tests with a fake model, no inference dependencies."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core_tools.server import CoreServer
from orchestration_tools.contracts import ManagerTurn
from orchestration_tools.manager import NativeManager


class FakeManager:
    model_name = 'gemma-4-E4B-it.litertlm'
    base_kv_tokens = 323

    def __init__(self):
        self.history = []
        self.events = []
        self.active = 0
        self.maximum_concurrent = 0
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.wait = False
        self.closed = False
        self.fail_next = False

    def respond(self, session_id, text):
        self.active += 1
        self.maximum_concurrent = max(self.active, self.maximum_concurrent)
        try:
            self.entered.set()
            if self.wait:
                if not self.resume.wait(4):
                    raise TimeoutError('fake model timeout')
            self.events.append(('user', session_id, text))
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError('inference failed')
            self.history.append(text)
            return ManagerTurn('reply', '/'.join(self.history))
        finally:
            self.active -= 1

    def release_session(self, session_id):
        self.events.append(('reset', session_id))
        self.history.clear()

    def close(self):
        self.closed = True


class CoreManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager = FakeManager()
        self.server = CoreServer(Path(self.tmp.name) / 'tools.sock', manager=self.manager)
        self.server.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(str(self.server.path))
        self.addCleanup(sock.close)
        reader = sock.makefile('r', encoding='utf-8')
        self.addCleanup(reader.close)
        return sock, reader

    @staticmethod
    def query(sock, payload):
        sock.sendall((json.dumps(payload, ensure_ascii=False)+'\n').encode())

    @staticmethod
    def recv(reader):
        return json.loads(reader.readline())

    def test_round_trip_native_manager_response(self):
        sock, r = self.connect()
        self.query(sock, {'type': 'acquire', 'client': 'tui'})
        acquired = self.recv(r)
        self.assertEqual(acquired['type'], 'acquired')
        self.assertEqual(acquired['core']['stage'], 2)
        self.assertEqual(acquired['core']['backend'], 'CPU')
        self.assertEqual(acquired['core']['manager_warm_tokens'], 323)
        self.assertTrue(acquired['status']['ready'])
        for text, expected in [('привет', 'привет'), ('выполни ls', 'привет/выполни ls')]:
            self.query(sock, {'type': 'user', 'text': text})
            accepted = self.recv(r)
            reply = self.recv(r)
            done = self.recv(r)
            self.assertEqual([accepted['type'],reply['type'],done['type']], ['accepted','reply','completed'])
            self.assertEqual(accepted['request_id'], reply['request_id'])
            self.assertEqual(reply['request_id'], done['request_id'])
            self.assertEqual(reply['text'], expected)
        self.assertEqual(self.manager.maximum_concurrent, 1)
        self.assertEqual(self.recv_after_query(sock,r,{'type':'snapshot'})['status']['state'], 'IDLE')

    def recv_after_query(self, sock, reader, payload):
        self.query(sock,payload)
        return self.recv(reader)

    def test_native_user_manager_transcript_is_logged(self):
        with self.assertLogs("agent_core_tools.server", level="INFO") as captured:
            sock, reader = self.connect()
            self.recv_after_query(sock, reader, {"type": "acquire", "client": "tui"})
            self.query(sock, {"type": "user", "text": "температура на улице"})
            self.assertEqual(self.recv(reader)["type"], "accepted")
            self.assertEqual(self.recv(reader)["type"], "reply")
            self.assertEqual(self.recv(reader)["type"], "completed")
        transcript = "\\n".join(captured.output)
        self.assertIn("USER -> MANAGER 'температура на улице'", transcript)
        self.assertIn("MANAGER -> USER kind=reply text='температура на улице'", transcript)

    def test_acquire_reports_selected_gpu_backend(self):
        self.manager.backend_name = 'GPU'
        sock, reader = self.connect()
        acquired = self.recv_after_query(sock, reader, {'type': 'acquire', 'client': 'tui'})
        self.assertEqual(acquired['core']['backend'], 'GPU')

    def test_release_resets_context_before_next_owner(self):
        a, r1 = self.connect()
        b, r2 = self.connect()
        self.assertEqual(self.recv_after_query(a,r1,{'type':'acquire','client':'a'})['type'],'acquired')
        self.query(a,{'type':'user','text':'secret'})
        for _ in range(3): self.recv(r1)
        self.assertEqual(self.recv_after_query(a,r1,{'type':'release'})['type'],'released')
        self.assertEqual(self.recv_after_query(b,r2,{'type':'acquire','client':'b'})['type'],'acquired')
        self.query(b,{'type':'user','text':'new'})
        self.assertEqual(self.recv(r2)['type'], 'accepted')
        self.assertEqual(self.recv(r2)['text'], 'new')
        self.assertEqual(self.recv(r2)['type'], 'completed')
        self.assertEqual([e[0] for e in self.manager.events], ['user','reset','user'])

    def test_stale_inflight_reply_not_delivered_to_new_owner(self):
        self.manager.wait = True
        a, r1 = self.connect()
        b, r2 = self.connect()
        self.recv_after_query(a,r1,{'type':'acquire','client':'a'})
        self.query(a,{'type':'user','text':'old'})
        self.assertEqual(self.recv(r1)['type'],'accepted')
        self.assertTrue(self.manager.entered.wait(1))
        self.assertEqual(self.recv_after_query(a,r1,{'type':'release'})['type'],'released')
        self.recv_after_query(b,r2,{'type':'acquire','client':'b'})
        self.query(b,{'type':'user','text':'new'})
        self.assertEqual(self.recv(r2)['type'], 'accepted')
        self.manager.resume.set()
        self.assertEqual(self.recv(r2)['text'], 'new')
        self.assertEqual(self.recv(r2)['type'], 'completed')
        self.assertEqual([e[0] for e in self.manager.events], ['user','reset','user'])
        self.assertEqual(self.manager.maximum_concurrent, 1)

    def test_failed_inference_reports_error_and_resets(self):
        a, r = self.connect()
        self.recv_after_query(a,r,{'type':'acquire','client':'tui'})
        self.manager.fail_next = True
        self.query(a, {'type':'user','text':'fail'})
        self.assertEqual(self.recv(r)['type'],'accepted')
        err = self.recv(r)
        self.assertEqual(err['type'],'error')
        self.assertIn('inference failed',err['text'])
        self.assertEqual(self.recv(r)['type'],'completed')
        self.query(a, {'type':'user','text':'works'})
        self.assertEqual(self.recv(r)['type'],'accepted')
        self.assertEqual(self.recv(r)['text'],'works')
        self.assertEqual(self.recv(r)['type'],'completed')

    def test_disconnect_queue_resets_before_new_owner(self):
        a, r = self.connect()
        self.recv_after_query(a,r,{'type':'acquire','client':'a'})
        self.query(a,{'type':'user','text':'old'})
        for _ in range(3): self.recv(r)
        a.shutdown(socket.SHUT_RDWR)
        a.close()
        b, r2 = self.connect()
        for _ in range(100):
            if self.recv_after_query(b,r2,{'type':'who'})['owner'] is None:
                break
            time.sleep(0.01)
        self.recv_after_query(b,r2,{'type':'acquire','client':'b'})
        self.query(b,{'type':'user','text':'new'})
        for _ in range(1): self.assertEqual(self.recv(r2)['type'], 'accepted')
        self.assertEqual(self.recv(r2)['text'], 'new')
        self.assertEqual(self.recv(r2)['type'], 'completed')


class NativeManagerTests(unittest.TestCase):
    def test_native_manager_wraps_conversation_without_legacy_message_logic(self):
        class FakeChat:
            base_kv_tokens = 250
            def __init__(self):
                self.questions=[]
                self.reset_count=0
                self.closed=False
            def ask(self, text):
                self.questions.append(text)
                return 'result' if text != 'silent' else ''
            def new_dialogue(self): self.reset_count+=1
            def close(self): self.closed=True
        class Engine:
            def __init__(self): self.closed=False
            def close(self): self.closed=True
        chat, engine=FakeChat(),Engine()
        manager=NativeManager(chat,engine,model_name='gemma')
        self.assertEqual(manager.respond('session','shell ls').text,'result')
        self.assertEqual(manager.respond('session','silent').kind,'silent')
        manager.release_session('session')
        self.assertEqual(chat.reset_count,1)
        manager.close()
        self.assertTrue(chat.closed and engine.closed)


if __name__=='__main__': unittest.main()
