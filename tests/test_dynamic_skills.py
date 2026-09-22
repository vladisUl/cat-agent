from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.dynamic_skills import load_dynamic_skills
from orchestration.skills import SkillBase, SkillBaseError
from orchestration.tool_catalog import ToolCatalog


BASE = """[TOOL shell]
name: shell
description:
Shell
prompt:
Use shell.
[/TOOL]
"""


class DynamicSkillsTest(unittest.TestCase):
    def test_loads_skill_from_filename_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            skills = root / "skills"
            skills.mkdir()
            (skills / "prognoz.txt").write_text(
                "code: /work#prognoz.sh\n"
                "description: прогноз на 3 дня в городе\n"
                "manager: true\n",
                encoding="utf-8",
            )

            loaded = load_dynamic_skills(skills)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "prognoz")
            self.assertEqual(loaded[0].code, "/work#prognoz.sh")
            self.assertEqual(loaded[0].description, "прогноз на 3 дня в городе")
            self.assertTrue(loaded[0].manager)
            self.assertIn("/work#prognoz.sh", loaded[0].as_skill().prompt)

    def test_manager_false_is_assignable_but_not_direct(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "prompt_base.txt"
            prompt.write_text(BASE, encoding="utf-8")
            skills = root / "skills"
            skills.mkdir()
            (skills / "background.txt").write_text(
                "code: /work#background.sh\n"
                "description: фоновая обработка\n"
                "manager: false\n",
                encoding="utf-8",
            )
            spec = load_dynamic_skills(skills)
            catalog = ToolCatalog(SkillBase(prompt), spec)

            self.assertIn("background", catalog.names())
            self.assertNotIn("background", catalog.manager_names())
            self.assertEqual(catalog.require(("background",))[0].name, "background")
            self.assertIn("background — фоновая обработка", catalog.catalog_text())
            self.assertIn("background — фоновая обработка [agent-only]", catalog.dynamic_prompt())
            self.assertNotIn("/work#background.sh", catalog.dynamic_prompt())

    def test_manager_true_gets_direct_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "prompt_base.txt"
            prompt.write_text(BASE, encoding="utf-8")
            skills = root / "skills"
            skills.mkdir()
            (skills / "prognoz.txt").write_text(
                "code: /work#prognoz.sh\n"
                "description: прогноз\n"
                "manager: true\n",
                encoding="utf-8",
            )
            spec = load_dynamic_skills(skills)
            catalog = ToolCatalog(SkillBase(prompt), spec)

            self.assertIn("prognoz", catalog.manager_names())
            self.assertIn("prognoz — прогноз [direct+agent]", catalog.dynamic_prompt())
            self.assertIn("Использование: /work#prognoz.sh", catalog.dynamic_prompt())

    def test_rejects_reserved_command_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            skills = Path(temp)
            (skills / "timer.txt").write_text(
                "code: /work#timer.sh\n"
                "description: collision\n"
                "manager: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillBaseError, "Reserved"):
                load_dynamic_skills(skills)

    def test_rejects_bad_code_and_manager_value(self) -> None:
        cases = (
            (
                "code: /work#other.sh\ndescription: x\nmanager: true\n",
                "code must be exactly",
            ),
            (
                "code: /work#demo.sh\ndescription: x\nmanager: maybe\n",
                "manager must be true or false",
            ),
        )
        for content, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                skills = Path(temp)
                (skills / "demo.txt").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(SkillBaseError, message):
                    load_dynamic_skills(skills)


if __name__ == "__main__":
    unittest.main()
