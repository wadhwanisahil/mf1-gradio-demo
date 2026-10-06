from __future__ import annotations

import unittest

from mf_demo.backend import MockBackend
from mf_demo.ui import build_demo, css, launch_style


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = build_demo(MockBackend())
        cls.config = cls.demo.get_config_file()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.demo.close()

    def test_ui_has_expected_api_endpoints(self) -> None:
        names = {
            dependency.get("api_name")
            for dependency in self.config["dependencies"]
            if dependency.get("api_name")
        }
        self.assertTrue(
            {
                "generate_image",
                "generate_unconditional",
                "analyze_image",
                "continue_text",
            }.issubset(names)
        )

    def test_ui_contains_research_tasks(self) -> None:
        labels = {
            component.get("props", {}).get("label") for component in self.config["components"]
        }
        self.assertIn("MF-1 output", labels)
        self.assertIn("MF-1 response", labels)
        self.assertIn("MF-1 continuation", labels)

    def test_style_is_loaded_at_launch(self) -> None:
        stylesheet = css()
        self.assertIn(".mf-title", stylesheet)
        self.assertIn("prefers-reduced-motion", stylesheet)
        style = launch_style()
        self.assertIn("theme", style)
        self.assertIn("css", style)
        self.assertIn("allowed_paths", style)


if __name__ == "__main__":
    unittest.main()
