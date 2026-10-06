"""Numerical and placement checks for the opt-in compatibility path."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RuntimeTests(unittest.TestCase):
    def test_continuation_accepts_unset_optional_stop_field(self) -> None:
        from mf_demo.backend import MFBackend, RuntimeOptions

        backend = MFBackend(RuntimeOptions(Path("/unused/MF/sft"), Path("/unused/assets")))
        received = []

        def complete_text(prompts, *, target_length, stop, config):
            received.append(stop)
            return ["a valid continuation"]

        pipeline = SimpleNamespace(complete_text=complete_text)
        with (
            patch.object(backend, "_prepare_task", return_value=pipeline),
            patch.object(backend, "_metadata", return_value={}),
        ):
            for stop in (None, "", "  END  \n\n STOP "):
                result, _ = backend.continue_text("A model can", 64, 16, 2, 42, stop)
                self.assertEqual(result, "a valid continuation")
        self.assertEqual(received, [(), (), ("END", "STOP")])

    def test_sdpa_preserves_official_noisy_text_blocks(self) -> None:
        from mf.modeling.attention import _run_masked_sdpa
        from mf.modeling.chunk_adapter import build_chunk_flex_prewarm_mask

        mask = build_chunk_flex_prewarm_mask(
            torch.device("cpu"),
            sequence_length=128,
            kernel_block_size=128,
            text_block_size=8,
        )
        generator = torch.Generator().manual_seed(9)
        q, k, v = [torch.randn(1, 2, 128, 8, generator=generator) for _ in range(3)]
        actual = _run_masked_sdpa(q, k, v, mask)
        block = torch.arange(128) // 8
        allowed = block[:, None] == block[None, :]
        scores = q @ k.transpose(-1, -2) / (8**0.5)
        expected = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
        torch.testing.assert_close(actual, expected)

    def test_sdpa_preserves_sequence_and_chunk_visibility(self) -> None:
        from mf.modeling.attention import _run_masked_sdpa

        generator = torch.Generator().manual_seed(42)
        q, k, v = [torch.randn(1, 2, 6, 8, generator=generator) for _ in range(3)]
        sequence = torch.tensor([0, 0, 1, 1, 1, 1])
        chunk = torch.tensor([0, 1, 0, 0, 1, 1])

        def mask_mod(batch, head, query, key):
            return (sequence[query] == sequence[key]) & (chunk[key] <= chunk[query])

        block_mask = SimpleNamespace(mask_mod=mask_mod)
        actual = _run_masked_sdpa(q, k, v, block_mask)
        allowed = mask_mod(0, 0, torch.arange(6)[:, None], torch.arange(6)[None, :])
        scores = (q @ k.transpose(-1, -2)) / (8**0.5)
        expected = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
        torch.testing.assert_close(actual, expected)
        # Perturb another sequence: the first sequence must remain unaffected.
        changed = v.clone()
        changed[:, :, 2:] += 100
        isolated = _run_masked_sdpa(q, k, changed, block_mask)
        torch.testing.assert_close(actual[:, :, :2], isolated[:, :, :2])

    def test_non_finite_prediction_is_rejected(self) -> None:
        from mf.evaluation.sampling import _prediction

        expected = torch.zeros(1, 2, 3)
        output = SimpleNamespace(text_x0=torch.full_like(expected, float("nan")))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _prediction(output, "text_x0", expected)

    def test_non_finite_image_is_rejected_before_clamping(self) -> None:
        from mf_demo.backend import BackendError, tensor_to_pil

        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(BackendError):
                tensor_to_pil(torch.full((3, 2, 2), value))

    def test_codec_roundtrip_between_gpus(self) -> None:
        from mf.codecs.placement import DeviceCodec

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("two CUDA GPUs are required")

        class EchoCodec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, device="cuda:1"))

            def encode(self, value, mask):
                self.assert_device(value)
                self.assert_device(mask)
                return value * self.weight

            def decode(self, value):
                self.assert_device(value)
                return value * self.weight

            def assert_device(self, value):
                if value.device != self.weight.device:
                    raise AssertionError("codec inputs were not transferred")

        codec = DeviceCodec(EchoCodec(), torch.device("cuda:1"))
        value = torch.arange(8, device="cuda:0", dtype=torch.float32)
        encoded = codec.encode(value, torch.ones_like(value, dtype=torch.bool))
        decoded = codec.decode(value)
        self.assertEqual(encoded.device, value.device)
        self.assertEqual(decoded.device, value.device)
        torch.testing.assert_close(encoded, value)
        torch.testing.assert_close(decoded, value)
