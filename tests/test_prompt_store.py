from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cat_agent.prompt_store import PromptStore
from cat_agent.skills import Skill


class PromptStoreTest(unittest.TestCase):
    def test_adds_matching_skill_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            (prompt_dir / "sys_prompt_agent_1.txt").write_text(
                "You are an agent.\n",
                encoding="utf-8",
            )
            (prompt_dir / "mqtt.txt").write_text(
                "broker: 192.168.0.21\ntopic: zigbee2mqtt/temp_ulica\n",
                encoding="utf-8",
            )
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="mqtt", description="MQTT", prompt="Use MQTT.")

            prompt = store.build_agent_prompt(
                "agent1",
                "Read outdoor temperature",
                (skill,),
                Path("/opt/model"),
            )

            self.assertIn("You are an agent.", prompt)
            self.assertIn('"workspace": "/opt/model"', prompt)
            self.assertIn('"name": "mqtt"', prompt)
            self.assertIn('"instructions": "Use MQTT."', prompt)
            self.assertIn("broker: 192.168.0.21", prompt)
            self.assertIn("topic: zigbee2mqtt/temp_ulica", prompt)

    def test_task_is_separate_from_stable_system_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            (prompt_dir / "sys_prompt_agent_1.txt").write_text(
                "You are an agent.\n",
                encoding="utf-8",
            )
            (prompt_dir / "mqtt.txt").write_text(
                "broker: 192.168.0.21\n",
                encoding="utf-8",
            )
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="mqtt", description="MQTT", prompt="Use MQTT.")

            bootstrap = store.build_agent_bootstrap((skill,), Path("/opt/model"))
            system_context = store.build_agent_system_context(
                "agent1", (skill,), Path("/opt/model")
            )
            task = store.build_agent_task("Read aquarium temperature")
            query = store.build_agent_task("Check user.txt", "query")
            context = store.build_agent_context("Имя файла: answer.txt")

            bootstrap_json = json.loads(bootstrap)
            self.assertEqual(bootstrap_json["workspace"], "/opt/model")
            self.assertEqual(bootstrap_json["skills"][0]["name"], "mqtt")
            self.assertEqual(bootstrap_json["skills"][0]["instructions"], "Use MQTT.")
            self.assertIn("broker: 192.168.0.21", bootstrap_json["skills"][0]["context"])
            self.assertNotIn("Read aquarium temperature", bootstrap)
            self.assertTrue(system_context.startswith("You are an agent."))

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

    def test_missing_skill_context_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            (prompt_dir / "sys_prompt_agent_1.txt").write_text(
                "You are an agent.\n",
                encoding="utf-8",
            )
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="shell", description="Shell", prompt="Use shell.")

            bootstrap = json.loads(store.build_agent_bootstrap((skill,), Path("/opt/model")))

            self.assertEqual(bootstrap["skills"][0]["name"], "shell")
            self.assertEqual(bootstrap["skills"][0]["instructions"], "Use shell.")
            self.assertNotIn("context", bootstrap["skills"][0])


if __name__ == "__main__":
    unittest.main()
