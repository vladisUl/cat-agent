from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestration.prompt_store import PromptStore
from orchestration.skills import Skill


MANAGER = """Manager base.

**СПИСОК TOOLS:**

After tools.

**Список топиков и полей mqtt:**

End.
"""

AGENT = "Agent base.\n"


class PromptStoreTest(unittest.TestCase):
    @staticmethod
    def _store(root: Path) -> PromptStore:
        (root / "sys_prompt_manager.md").write_text(MANAGER, encoding="utf-8")
        (root / "sys_prompt_agent_1.md").write_text(AGENT, encoding="utf-8")
        (root / "mqtt.md").write_text("MQTT TOPICS RAW\n", encoding="utf-8")
        store = PromptStore(root, agent_count=1)
        store.validate()
        return store

    def test_manager_inserts_tool_files_and_mqtt_at_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(Path(temp))
            shell_raw = (
                "**[TOOL shell]**\n"
                "**name:** shell\n"
                "**manager:** true\n"
                "**[/TOOL]**"
            )
            mqtt_raw = (
                "**[TOOL mqtt]**\n"
                "**name:** mqtt\n"
                "**manager:** true\n"
                "**[/TOOL]**"
            )
            tools = (
                Skill("shell", "shell", shell_raw),
                Skill("mqtt", "mqtt", mqtt_raw),
            )

            prompt = store.manager_system_prompt(tools)

            self.assertIn(
                "**СПИСОК TOOLS:**\n\n" + shell_raw + "\n\n" + mqtt_raw,
                prompt,
            )
            self.assertIn(
                "**Список топиков и полей mqtt:**\n\nMQTT TOPICS RAW",
                prompt,
            )

    def test_agent_gets_assigned_tool_blocks_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = self._store(Path(temp))
            raw = (
                "**[TOOL mqtt]**\n"
                "**name:** mqtt\n"
                "**code:** mqtt_sub.sh <topic> <field>\n"
                "**manager:** true\n"
                "**[/TOOL]**"
            )
            tool = Skill("mqtt", "mqtt", raw)

            bootstrap = store.build_agent_bootstrap((tool,), Path("/opt/model"))
            system_context = store.build_agent_system_context(
                "agent1", (tool,), Path("/opt/model")
            )

            self.assertIn("**WORKSPACE:** /opt/model", bootstrap)
            self.assertIn(raw, bootstrap)
            self.assertIn("MQTT TOPICS RAW", bootstrap)
            self.assertEqual(bootstrap.count(raw), 1)
            self.assertTrue(system_context.startswith("Agent base."))
            self.assertIn(raw, system_context)

    def test_task_and_context_protocol_remain_json(self) -> None:
        task = PromptStore.build_agent_task("Read aquarium temperature")
        query = PromptStore.build_agent_task("Check user.txt", "query")
        context = PromptStore.build_agent_context("Имя файла: answer.txt")

        self.assertEqual(
            json.loads(task),
            {"task": "Read aquarium temperature"},
        )
        self.assertEqual(
            json.loads(query),
            {"method": "query", "task": "Check user.txt"},
        )
        self.assertEqual(
            json.loads(context),
            {"context": "Имя файла: answer.txt"},
        )

    def test_validate_requires_unique_insertion_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sys_prompt_manager.md").write_text(
                "**СПИСОК TOOLS:**\n",
                encoding="utf-8",
            )
            (root / "sys_prompt_agent_1.md").write_text(AGENT, encoding="utf-8")
            (root / "mqtt.md").write_text("topics\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Список топиков"):
                PromptStore(root, 1).validate()

    def test_agent_prompt_is_not_rewritten_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            store = PromptStore(prompt_dir, agent_count=1)
            path = prompt_dir / "prompt_agent_1.md"
            path.write_text("stable prompt\n", encoding="utf-8")

            with patch.object(Path, "write_text", autospec=True) as write_text:
                store.write_agent_prompt("agent1", "stable prompt")
                write_text.assert_not_called()

                store.write_agent_prompt("agent1", "changed prompt")
                write_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
