from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PIL import Image

from mf_demo.backend import (
    BackendError,
    MAX_SEED,
    MockBackend,
    RuntimeOptions,
    choose_seed,
    normalize_seed,
    require_text,
)


class ValidationTests(unittest.TestCase):
    def test_text_validation(self) -> None:
        self.assertEqual(require_text("  hello  ", "Prompt"), "hello")
        with self.assertRaisesRegex(BackendError, "Prompt"):
            require_text("   ", "Prompt")

    def test_seed_validation(self) -> None:
        self.assertEqual(normalize_seed(42.9), 42)
        for value in (-1, MAX_SEED + 1, None):
            with self.subTest(value=value), self.assertRaises(BackendError):
                normalize_seed(value)
        self.assertTrue(0 <= choose_seed(0, True) <= MAX_SEED)

    def test_runtime_options_from_environment(self) -> None:
        environment = {
            "MF_CHECKPOINT": "/tmp/MF/sft",
            "MF_ASSETS_ROOT": "/tmp/assets",
            "MF_DEVICE": "cuda:1",
            "MF_WEIGHTS": "raw",
            "MF_RELEASE_CODECS_ON_TASK_SWITCH": "0",
        }
        with patch.dict(os.environ, environment, clear=True):
            options = RuntimeOptions.from_env()
        self.assertEqual(options.device, "cuda:1")
        self.assertEqual(options.weights, "raw")
        self.assertFalse(options.release_codecs_on_task_switch)

    def test_runtime_options_require_paths(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(BackendError):
            RuntimeOptions.from_env()

    def test_compatibility_options_are_explicit(self) -> None:
        environment = {
            "MF_CHECKPOINT": "/tmp/MF/sft",
            "MF_ASSETS_ROOT": "/tmp/assets",
            "MF_PRECISION": "fp16",
            "MF_CODEC_DEVICE": "cuda:1",
            "MF_ATTENTION_BACKEND": "sdpa",
            "MF_ALLOW_LOW_VRAM": "1",
        }
        with patch.dict(os.environ, environment, clear=True):
            options = RuntimeOptions.from_env()
        self.assertEqual(options.precision, "fp16")
        self.assertEqual(options.codec_device, "cuda:1")
        self.assertEqual(options.attention_backend, "sdpa")
        self.assertTrue(options.allow_low_vram)
        with patch.dict(os.environ, {**environment, "MF_PRECISION": "fp8"}, clear=True):
            with self.assertRaises(BackendError):
                RuntimeOptions.from_env()


class MockBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockBackend()

    def test_images_are_reproducible(self) -> None:
        first, first_meta = self.backend.generate_image("A lighthouse", 16, 2, "sde", 7)
        second, second_meta = self.backend.generate_image("A lighthouse", 16, 2, "sde", 7)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_meta, second_meta)
        self.assertTrue(first_meta["mock"])

    def test_caption_and_text(self) -> None:
        image = Image.new("RGB", (30, 20), "navy")
        caption, metadata = self.backend.caption(image, "What is shown?", 64, 16, 2, 9)
        self.assertIn("30×20", caption)
        self.assertEqual(metadata["seed"], 9)
        continuation, metadata = self.backend.continue_text("A model", 64, 16, 2, 4)
        self.assertTrue(continuation.startswith("A model"))
        self.assertEqual(metadata["task"], "text-continuation")

    def test_caption_requires_image(self) -> None:
        with self.assertRaisesRegex(BackendError, "Upload"):
            self.backend.caption(None, "Describe", 64, 16, 2, 1)


if __name__ == "__main__":
    unittest.main()
