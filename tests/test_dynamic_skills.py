from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.tool_catalog import build_tool_catalog


class LegacySkillsDirectoryTest(unittest.TestCase):
    def test_skills_directory_is_not_a_tool_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = root / "tools"
            tools.mkdir()
            (tools / "demo.md").write_text(
                "**[TOOL demo]**\n"
                "**name:** demo\n"
                "**description:** canonical tool\n"
                "**manager:** true\n"
                "**[/TOOL]**\n",
                encoding="utf-8",
            )

            skills = root / "skills"
            skills.mkdir()
            (skills / "ghost.txt").write_text(
                "code: /work#ghost.sh\n"
                "description: must be ignored\n"
                "manager: true\n",
                encoding="utf-8",
            )

            catalog = build_tool_catalog(tools, ())

            self.assertIn("demo", catalog.names())
            self.assertNotIn("ghost", catalog.names())
            self.assertEqual(catalog.manager_names(), ("demo",))


if __name__ == "__main__":
    unittest.main()
