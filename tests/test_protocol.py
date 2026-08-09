from __future__ import annotations

import unittest

from cat_agent.protocol import (
    AgentAction,
    ManagerAction,
    parse_agent_output,
    parse_manager_output,
)


class ProtocolTest(unittest.TestCase):
    def test_manager_delegate(self) -> None:
        item = parse_manager_output("DELEGATE shell,mqtt\nСделай задачу")
        self.assertEqual(item.action, ManagerAction.DELEGATE)
        self.assertEqual(item.skills, ("shell", "mqtt"))
        self.assertEqual(item.body, "Сделай задачу")

    def test_manager_continue(self) -> None:
        item = parse_manager_output("CONTINUE agent2\nbroker=127.0.0.1")
        self.assertEqual(item.action, ManagerAction.CONTINUE)
        self.assertEqual(item.agent_id, "agent2")

    def test_agent_done_need_and_command(self) -> None:
        done = parse_agent_output("DONE\nГотово")
        self.assertEqual(done.action, AgentAction.DONE)
        self.assertEqual(done.body, "Готово")
        need = parse_agent_output("NEED\nНужен topic")
        self.assertEqual(need.action, AgentAction.NEED)
        command = parse_agent_output("ls -l")
        self.assertEqual(command.action, AgentAction.COMMAND)
        self.assertEqual(command.command, "ls -l")

    def test_rejects_bad_shapes(self) -> None:
        self.assertIsNotNone(parse_manager_output("hello").error)
        self.assertIsNotNone(parse_agent_output("ls\ncat file").error)
        self.assertIsNotNone(parse_agent_output("DONE").error)


if __name__ == "__main__":
    unittest.main()
