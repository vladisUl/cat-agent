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

    def test_manager_delegate_rejects_duplicate_skills(self) -> None:
        item = parse_manager_output("DELEGATE mqtt,mqtt\nПолучи температуру")
        self.assertIsNone(item.action)
        self.assertEqual(item.error, "DELEGATE contains duplicate skills")

    def test_manager_continue(self) -> None:
        item = parse_manager_output("CONTINUE agent2\nbroker=127.0.0.1")
        self.assertEqual(item.action, ManagerAction.CONTINUE)
        self.assertEqual(item.agent_id, "agent2")

    def test_manager_system_timer_set(self) -> None:
        item = parse_manager_output(
            "SYSTEM TIMER SET cats 60\nПроверяй test.txt и обрабатывай содержимое."
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TIMER SET cats 60")
        self.assertEqual(item.body, "Проверяй test.txt и обрабатывай содержимое.")

    def test_manager_system_rejects_embedded_control_action(self) -> None:
        item = parse_manager_output(
            "SYSTEM TIMER SET cats 60\n"
            "DELEGATE shell\n"
            "Проверь содержимое test.txt."
        )
        self.assertIsNone(item.action)
        self.assertIn("must contain only the future event task", item.error or "")

        item = parse_manager_output(
            "SYSTEM TIMER STOP cats\n"
            "REPLY\n"
            "Таймер остановлен."
        )
        self.assertIsNone(item.action)
        self.assertIn("must not contain additional text", item.error or "")

    def test_agent_done_need_and_command(self) -> None:
        done = parse_agent_output("DONE\nГотово")
        self.assertEqual(done.action, AgentAction.DONE)
        self.assertEqual(done.body, "Готово")
        need = parse_agent_output("NEED\nНужен topic")
        self.assertEqual(need.action, AgentAction.NEED)
        command = parse_agent_output("ls -l")
        self.assertEqual(command.action, AgentAction.COMMAND)
        self.assertEqual(command.command, "ls -l")

    def test_agent_rejects_multiple_actions(self) -> None:
        two_commands = parse_agent_output("test -e test.txt\ncat test.txt")
        self.assertIsNone(two_commands.action)
        self.assertIn("no command was executed", two_commands.error or "")

        command_and_need = parse_agent_output(
            "cat test.txt\nNEED\nФайл test.txt не существует."
        )
        self.assertIsNone(command_and_need.action)
        self.assertIn("no command was executed", command_and_need.error or "")

    def test_rejects_bad_shapes(self) -> None:
        self.assertIsNotNone(parse_manager_output("hello").error)
        self.assertIsNotNone(parse_agent_output("ls\ncat file").error)
        self.assertIsNotNone(parse_agent_output("DONE").error)


if __name__ == "__main__":
    unittest.main()
