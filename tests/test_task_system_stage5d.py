"""Stage 5D TASK SYSTEM recovery never replays uncertain native side effects."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from task_system.engine import TaskEngine


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.live = {"old", "new"}
        self.engine = TaskEngine(
            root / "state.db", task_file=root / "tasks.txt",
            event_file=root / "events.json", alive=lambda peer: peer in self.live,
        )
        self.addCleanup(self.engine.close)

    def call(self, target, method, *, peer="old", **kw):
        return self.engine.call(target, method, kwargs=kw, peer=peer)

    def create_run(self, method="query"):
        task = self.engine.runtime.create_periodic_task(
            "recovery", "TEST_SIDE_EFFECT", ("shell",), 3600,
            method=method, executor="litert",
        )
        run_id = self.engine.call("system", "run_task", (task.task_id,))
        return task, run_id

    def offer(self, *, peer="old", owner="owner-old"):
        return self.call(
            "worker", "pull", peer=peer,
            owner=owner, executor="litert", seen=[],
        )

    def begin(self, event):
        return self.call(
            "worker", "begin", owner="owner-old",
            run_id=event["run_id"], offer_id=event["offer_id"],
        )

    def test_worker_crash_while_running_is_interrupted_not_reoffered(self):
        _, run_id = self.create_run()
        offer = self.offer()
        self.assertTrue(self.begin(offer))
        self.live.remove("old")
        self.engine.tick()
        run = self.engine.runs[run_id]
        self.assertEqual(run["state"], "interrupted")
        self.assertEqual(run["owner"], "")
        self.assertIsNone(self.offer(peer="new", owner="owner-new"))
        self.assertEqual(self.engine.runs[run_id]["state"], "interrupted")

    def test_unstarted_offer_crash_is_requeued_with_new_offer_id(self):
        _, run_id = self.create_run()
        first = self.offer()
        self.live.remove("old")
        self.engine.tick()
        self.assertEqual(self.engine.runs[run_id]["state"], "pending")
        other = self.offer(peer="new", owner="owner-new")
        self.assertEqual(other["run_id"], run_id)
        self.assertNotEqual(other["offer_id"], first["offer_id"])
        self.assertEqual(self.engine.runs[run_id]["state"], "offered")

    def test_worker_crash_after_sqlite_commit_preserves_unread_query(self):
        task, run_id = self.create_run()
        offer = self.offer()
        self.assertTrue(self.begin(offer))
        notice = {
            "type": "task_query_result", "run_id": run_id,
            "task_id": task.task_id, "source": offer["source"],
            "agent_id": "agent_1", "agent_ok": True,
            "text": "Ответ после записи, до FINISH",
        }
        self.assertTrue(self.call(
            "worker", "publish_result", owner="owner-old",
            run_id=run_id, offer_id=offer["offer_id"], result=notice,
        ))
        self.live.remove("old")
        self.engine.tick()
        self.assertEqual(self.engine.runs[run_id]["state"], "interrupted")
        self.assertEqual(self.engine.call("notifications", "list"), [notice])
        self.assertIsNone(self.offer(peer="new", owner="owner-new"))
        self.assertTrue(self.engine.call(
            "notifications", "ack", (run_id,),
        ))
        self.assertEqual(self.engine.call("notifications", "list"), [])


if __name__ == "__main__":
    unittest.main()
