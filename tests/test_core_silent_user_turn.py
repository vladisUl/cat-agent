from __future__ import annotations

from concurrent.futures import Future
import unittest

from litert_agent.core_scheduler import CoreScheduler, MANAGER_PRIORITY, _PriorityRequest
from orchestration.manager import ManagerTurn


class CoreSilentUserTurnTest(unittest.TestCase):
    def _scheduler(self, delivered: list[ManagerTurn]) -> CoreScheduler:
        scheduler = CoreScheduler(object(), on_human_turn=delivered.append)
        scheduler._active_request = _PriorityRequest(
            kind="user",
            label="user",
            payload="test",
            queued_at=0.0,
            priority=MANAGER_PRIORITY,
        )
        return scheduler

    def test_silent_user_turn_is_not_delivered_to_interface(self) -> None:
        delivered: list[ManagerTurn] = []
        scheduler = self._scheduler(delivered)
        future: Future[ManagerTurn] = Future()
        future.set_result(ManagerTurn("silent", ""))

        try:
            scheduler._finish_regular_request(future)
            self.assertEqual(delivered, [])
        finally:
            scheduler._executor.shutdown(wait=True, cancel_futures=False)

    def test_reply_user_turn_is_delivered_normally(self) -> None:
        delivered: list[ManagerTurn] = []
        scheduler = self._scheduler(delivered)
        future: Future[ManagerTurn] = Future()
        turn = ManagerTurn("reply", "OK")
        future.set_result(turn)

        try:
            scheduler._finish_regular_request(future)
            self.assertEqual(delivered, [turn])
        finally:
            scheduler._executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    unittest.main()
