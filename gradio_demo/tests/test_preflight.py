from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.preflight import _file_check


class PreflightTests(unittest.TestCase):
    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            check = _file_check("weights", Path(directory) / "missing.bin", 10)
        self.assertEqual(check.level, "FAIL")
        self.assertIn("missing", check.detail)

    def test_small_file_is_detected_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"1234")
            check = _file_check("weights", path, 10)
        self.assertEqual(check.level, "FAIL")
        self.assertIn("incomplete", check.detail)

    def test_valid_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"1234567890")
            check = _file_check("weights", path, 10)
        self.assertEqual(check.level, "PASS")


if __name__ == "__main__":
    unittest.main()
