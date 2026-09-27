from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.skills import SkillBaseError
from orchestration.tool_catalog import ToolCatalog, load_tool_files


ROOT = Path(__file__).resolve().parents[1]


class MarkdownToolCatalogTest(unittest.TestCase):
    def test_repository_tools_are_the_canonical_catalog(self) -> None:
        loaded = load_tool_files(ROOT / "tools")
        names = tuple(tool.name for tool in loaded)

        self.assertEqual(
            names,
            (
                "cyclic_process",
                "file_divide",
                "mqtt",
                "prognoz",
                "read_pic",
                "shell",
            ),
        )
        self.assertTrue(all(tool.manager for tool in loaded))
        for tool in loaded:
            self.assertEqual(
                tool.text,
                (ROOT / "tools" / f"{tool.name}.md")
                .read_text(encoding="utf-8")
                .strip(),
            )

    @staticmethod
    def _write_tool(
        directory: Path,
        name: str,
        *,
        manager: bool,
        body: str = "",
    ) -> str:
        text = (
            f"**[TOOL {name}]**\n\n"
            f"**name:** {name}\n\n"
            f"**description:** {name} description\n\n"
            f"{body}"
            f"**manager:** {'true' if manager else 'false'}\n\n"
            "**[/TOOL]**\n"
        )
        (directory / f"{name}.md").write_text(text, encoding="utf-8")
        return text.strip()

    def test_loads_raw_tool_text_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self._write_tool(root, "demo", manager=True, body="RAW BODY\n\n")

            loaded = load_tool_files(root)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "demo")
            self.assertTrue(loaded[0].manager)
            self.assertEqual(loaded[0].text, raw)
            self.assertEqual(loaded[0].as_skill().prompt, raw)

    def test_manager_flag_only_filters_direct_manager_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_tool(root, "direct", manager=True)
            self._write_tool(root, "agent_only", manager=False)

            catalog = ToolCatalog(load_tool_files(root))

            self.assertIn("direct", catalog.names())
            self.assertIn("agent_only", catalog.names())
            self.assertIn("direct", catalog.manager_names())
            self.assertNotIn("agent_only", catalog.manager_names())
            self.assertEqual(catalog.require(("agent_only",))[0].name, "agent_only")

    def test_rejects_filename_header_or_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "right.md").write_text(
                "**[TOOL wrong]**\n"
                "**name:** wrong\n"
                "**manager:** true\n"
                "**[/TOOL]**\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillBaseError, "header mismatch"):
                load_tool_files(root)

    def test_rejects_invalid_manager_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "demo.md").write_text(
                "**[TOOL demo]**\n"
                "**name:** demo\n"
                "**manager:** maybe\n"
                "**[/TOOL]**\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillBaseError, "manager must be true or false"):
                load_tool_files(root)


if __name__ == "__main__":
    unittest.main()
