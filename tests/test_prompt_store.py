from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cat_agent.prompt_store import PromptStore
from cat_agent.skills import Skill


class PromptStoreTest(unittest.TestCase):
    def test_adds_matching_skill_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
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

            self.assertIn("[SKILL mqtt]\nUse MQTT.\n[/SKILL]", prompt)
            self.assertIn("[CONTEXT mqtt]", prompt)
            self.assertIn("broker: 192.168.0.21", prompt)
            self.assertIn("topic: zigbee2mqtt/temp_ulica", prompt)

    def test_task_is_after_stable_skill_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            (prompt_dir / "mqtt.txt").write_text(
                "broker: 192.168.0.21\n",
                encoding="utf-8",
            )
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="mqtt", description="MQTT", prompt="Use MQTT.")

            prompt = store.build_agent_prompt(
                "agent1",
                "Read aquarium temperature",
                (skill,),
                Path("/opt/model"),
            )

            self.assertLess(prompt.index("[WORKSPACE]"), prompt.index("[SKILL mqtt]"))
            self.assertLess(prompt.index("[SKILL mqtt]"), prompt.index("[CONTEXT mqtt]"))
            self.assertLess(prompt.index("[CONTEXT mqtt]"), prompt.index("[TASK]"))
            self.assertTrue(prompt.rstrip().endswith("[/TASK]"))

    def test_missing_skill_context_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="shell", description="Shell", prompt="Use shell.")

            prompt = store.build_agent_prompt(
                "agent1",
                "List files",
                (skill,),
                Path("/opt/model"),
            )

            self.assertIn("[SKILL shell]\nUse shell.\n[/SKILL]", prompt)
            self.assertNotIn("[CONTEXT shell]", prompt)


if __name__ == "__main__":
    unittest.main()
