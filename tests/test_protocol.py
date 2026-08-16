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

    def test_manager_persistent_task_timer(self) -> None:
        item = parse_manager_output(
            'task_timer.sh 60 shell,mqtt "ловля котов"\nПроверять котов.'
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TASK TIMER SET 60")
        self.assertEqual(item.skills, ("shell", "mqtt"))
        self.assertEqual(item.task_description, "ловля котов")
        self.assertEqual(item.task_method, "task")
        self.assertEqual(item.body, "Проверять котов.")

    def test_manager_persistent_query_timer(self) -> None:
        item = parse_manager_output(
            'query_timer.sh 60 shell "проверка файла"\n'
            'Проверить user.txt. Вернуть "ОК" или "Авария".'
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TASK TIMER SET 60")
        self.assertEqual(item.skills, ("shell",))
        self.assertEqual(item.task_description, "проверка файла")
        self.assertEqual(item.task_method, "query")
        self.assertIn("Вернуть", item.body)

    def test_old_persistent_timer_form_defaults_to_task(self) -> None:
        item = parse_manager_output(
            'timer.sh 60 shell,mqtt "ловля котов"\nПроверять котов.'
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TASK TIMER SET 60")
        self.assertEqual(item.task_method, "task")

    def test_manager_persistent_timer_task_control(self) -> None:
        stop = parse_manager_output("timer.sh stop 2")
        self.assertEqual(stop.system_command, "TASK TIMER STOP 2")

        start = parse_manager_output("timer.sh start 2")
        self.assertEqual(start.system_command, "TASK TIMER START 2")

        period = parse_manager_output("timer.sh period 120 2")
        self.assertEqual(period.system_command, "TASK TIMER PERIOD 2 120")

        delete = parse_manager_output("timer.sh delete 2")
        self.assertEqual(delete.system_command, "TASK TIMER DELETE 2")

        listing = parse_manager_output("timer.sh list")
        self.assertEqual(listing.system_command, "TASK TIMER LIST")

    def test_manager_legacy_timer_control_is_still_accepted(self) -> None:
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

    def test_manager_timer_script_rejects_embedded_control(self) -> None:
        item = parse_manager_output(
            'query_timer.sh 60 shell "проверка"\n'
            "DELEGATE shell\n"
            "Проверь содержимое test.txt."
        )
        self.assertIsNone(item.action)
        self.assertIn("future task", item.error or "")

    def test_manager_timer_script_rejects_bad_syntax(self) -> None:
        self.assertIsNotNone(parse_manager_output("task_timer.sh nope shell \"x\"\nРаботай").error)
        self.assertIsNotNone(parse_manager_output("query_timer.sh 60 shell\nРаботай").error)
        self.assertIsNotNone(parse_manager_output("timer.sh stop cats\nREPLY\nготово").error)

    def test_agent_json_result_need_done_and_command(self) -> None:
        result = parse_agent_output('{"result":"Готово"}')
        self.assertEqual(result.action, AgentAction.DONE)
        self.assertEqual(result.body, "Готово")

        done = parse_agent_output('{"done":true}')
        self.assertEqual(done.action, AgentAction.DONE)
        self.assertEqual(done.body, "")

        need = parse_agent_output('{"need":"Нужен topic"}')
        self.assertEqual(need.action, AgentAction.NEED)
        self.assertEqual(need.body, "Нужен topic")

        command = parse_agent_output("ls -l")
        self.assertEqual(command.action, AgentAction.COMMAND)
        self.assertEqual(command.command, "ls -l")

    def test_agent_accepts_pretty_json_result(self) -> None:
        item = parse_agent_output('{\n  "result": "ОК"\n}')
        self.assertEqual(item.action, AgentAction.DONE)
        self.assertEqual(item.body, "ОК")

    def test_agent_rejects_multiple_actions(self) -> None:
        two_commands = parse_agent_output("test -e test.txt\ncat test.txt")
        self.assertIsNone(two_commands.action)
        self.assertIn("no command was executed", two_commands.error or "")

        command_and_json = parse_agent_output(
            'cat test.txt\n{"result":"Файл прочитан"}'
        )
        self.assertIsNone(command_and_json.action)
        self.assertIn("no command was executed", command_and_json.error or "")

        mixed_json = parse_agent_output('{"result":"ОК","done":true}')
        self.assertIsNone(mixed_json.action)
        self.assertIn("exactly one of", mixed_json.error or "")

    def test_agent_rejects_bad_json_shapes(self) -> None:
        self.assertIsNotNone(parse_agent_output('{"result":').error)
        self.assertIsNotNone(parse_agent_output('{"done":false}').error)
        self.assertIsNotNone(parse_agent_output('{"need":""}').error)
        self.assertIsNotNone(parse_agent_output('["result","ОК"]').error)

    def test_rejects_bad_manager_shape(self) -> None:
        self.assertIsNotNone(parse_manager_output("hello").error)


if __name__ == "__main__":
    unittest.main()
