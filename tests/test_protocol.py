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

    def test_manager_timer_script_set_named(self) -> None:
        item = parse_manager_output(
            "timer.sh 60 cats\nПроверяй test.txt и обрабатывай содержимое."
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TIMER SET cats 60")
        self.assertEqual(item.body, "Проверяй test.txt и обрабатывай содержимое.")

    def test_manager_timer_script_set_default(self) -> None:
        item = parse_manager_output("timer.sh 60\nПроверяй test.txt.")
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TIMER SET default 60")
        self.assertEqual(item.body, "Проверяй test.txt.")

    def test_manager_timer_script_control(self) -> None:
        stop = parse_manager_output("timer.sh stop cats")
        self.assertEqual(stop.action, ManagerAction.SYSTEM)
        self.assertEqual(stop.system_command, "TIMER STOP cats")

        start_default = parse_manager_output("timer.sh start")
        self.assertEqual(start_default.system_command, "TIMER START default")

        period = parse_manager_output("timer.sh period 120 cats")
        self.assertEqual(period.system_command, "TIMER PERIOD cats 120")

        period_alt = parse_manager_output("timer.sh period cats 120")
        self.assertEqual(period_alt.system_command, "TIMER PERIOD cats 120")

        delete = parse_manager_output("timer.sh delete cats")
        self.assertEqual(delete.system_command, "TIMER DELETE cats")

        listing = parse_manager_output("timer.sh list")
        self.assertEqual(listing.system_command, "TIMER LIST")

    def test_manager_timer_script_rejects_embedded_control(self) -> None:
        item = parse_manager_output(
            "timer.sh 60 cats\n"
            "DELEGATE shell\n"
            "Проверь содержимое test.txt."
        )
        self.assertIsNone(item.action)
        self.assertIn("future event task", item.error or "")

    def test_manager_timer_script_rejects_bad_syntax(self) -> None:
        self.assertIsNotNone(parse_manager_output("timer.sh nope\nПроверяй файл").error)
        self.assertIsNotNone(parse_manager_output("timer.sh 60 cats extra\nПроверяй файл").error)
        self.assertIsNotNone(parse_manager_output("timer.sh stop cats\nREPLY\nготово").error)

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
