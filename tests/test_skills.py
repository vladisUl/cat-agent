from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.skills import SkillBase


class SkillBaseTest(unittest.TestCase):
    def test_parse_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompt_base.txt"
            path.write_text(
                """[SKILL shell]\nname: shell\ndescription:\nLinux\nprompt:\nUse shell.\n[/SKILL]\n""",
                encoding="utf-8",
            )
            base = SkillBase(path)
            self.assertEqual(base.names(), ("shell",))
            self.assertEqual(base.get("shell").prompt, "Use shell.")
            self.assertEqual(base.catalog_text(), "shell — Linux")


if __name__ == "__main__":
    unittest.main()
