from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cat_agent.prompt_store import AGENT_BOOTSTRAP_ACK, PromptStore
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

    def test_task_is_separate_from_stable_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            (prompt_dir / "mqtt.txt").write_text(
                "broker: 192.168.0.21\n",
                encoding="utf-8",
            )
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="mqtt", description="MQTT", prompt="Use MQTT.")

            bootstrap = store.build_agent_bootstrap((skill,), Path("/opt/model"))
            task = store.build_agent_task("Read aquarium temperature")
            query = store.build_agent_task("Check user.txt", "query")

            self.assertIn("[WORKSPACE]", bootstrap)
            self.assertIn("[SKILL mqtt]", bootstrap)
            self.assertIn("[CONTEXT mqtt]", bootstrap)
            self.assertIn("[BOOTSTRAP]", bootstrap)
            self.assertIn(AGENT_BOOTSTRAP_ACK, bootstrap)
            self.assertNotIn("[TASK]", bootstrap)
            self.assertNotIn("Read aquarium temperature", bootstrap)
            self.assertEqual(task, "[TASK]\nRead aquarium temperature\n[/TASK]\n")
            self.assertEqual(
                query,
                "[METHOD]\nQUERY\n[/METHOD]\n[TASK]\nCheck user.txt\n[/TASK]\n",
            )

    def test_missing_skill_context_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prompt_dir = Path(temp)
            store = PromptStore(prompt_dir, agent_count=1)
            skill = Skill(name="shell", description="Shell", prompt="Use shell.")

            bootstrap = store.build_agent_bootstrap((skill,), Path("/opt/model"))

            self.assertIn("[SKILL shell]\nUse shell.\n[/SKILL]", bootstrap)
            self.assertNotIn("[CONTEXT shell]", bootstrap)


if __name__ == "__main__":
    unittest.main()
