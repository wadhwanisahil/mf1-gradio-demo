"""Backends for the MF-1 Gradio application.

The real backend imports PyTorch and MF only when the model is loaded.  This
keeps mock mode and the UI testable on machines without CUDA or model assets.
"""

from __future__ import annotations

import gc
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MIN_SUPPORTED_VRAM_GIB = 10.0
MIN_RECOMMENDED_VRAM_GIB = 20.0
MAX_SEED = 2**31 - 1


class BackendError(RuntimeError):
    """A safe, user-facing backend error."""


def require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise BackendError(f"{label} is required.")
    return normalized


def normalize_seed(value: int | float | None) -> int:
    if value is None:
        raise BackendError("Seed is required.")
    seed = int(value)
    if seed < 0 or seed > MAX_SEED:
        raise BackendError(f"Seed must be between 0 and {MAX_SEED:,}.")
    return seed


def choose_seed(value: int | float | None, randomize: bool) -> int:
    return random.SystemRandom().randint(0, MAX_SEED) if randomize else normalize_seed(value)


def _integer(value: int | float, label: str, *, minimum: int, maximum: int) -> int:
    result = int(value)
    if result < minimum or result > maximum:
        raise BackendError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _number(value: int | float, label: str, *, minimum: float, maximum: float) -> float:
    result = float(value)
    if result < minimum or result > maximum:
        raise BackendError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return result


def tensor_to_pil(tensor: Any) -> Image.Image:
    """Convert one MF image tensor in ``[-1, 1]`` to an RGB PIL image."""

    pixels = tensor.detach().cpu().float()
    if not bool(pixels.isfinite().all()):
        raise BackendError("MF generated non-finite image values; check inference precision.")
    pixels = (pixels.clamp(-1, 1) + 1) * 127.5
    array = pixels.permute(1, 2, 0).numpy().round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    checkpoint: Path
    assets_root: Path
    device: str = "cuda:0"
    weights: str = "ema"
    release_codecs_on_task_switch: bool = True
    precision: str = "bf16"
    codec_device: str | None = None
    attention_backend: str = "flex"
    allow_low_vram: bool = False

    def __post_init__(self) -> None:
        if self.precision not in {"bf16", "fp16"}:
            raise BackendError("MF_PRECISION must be 'bf16' or 'fp16'.")
        if self.attention_backend not in {"flex", "sdpa"}:
            raise BackendError("MF_ATTENTION_BACKEND must be 'flex' or 'sdpa'.")

    @classmethod
    def from_env(cls) -> "RuntimeOptions":
        checkpoint = os.environ.get("MF_CHECKPOINT")
        assets_root = os.environ.get("MF_ASSETS_ROOT")
        if not checkpoint:
            raise BackendError(
                "MF_CHECKPOINT is not set. Point it to the released MF/sft directory."
            )
        if not assets_root:
            raise BackendError(
                "MF_ASSETS_ROOT is not set. Point it to the directory containing "
                "scale_rae_decoder/."
            )
        weights = os.environ.get("MF_WEIGHTS", "ema").lower()
        if weights not in {"ema", "raw"}:
            raise BackendError("MF_WEIGHTS must be either 'ema' or 'raw'.")
        release = os.environ.get("MF_RELEASE_CODECS_ON_TASK_SWITCH", "1") != "0"
        return cls(
            checkpoint=Path(checkpoint).expanduser(),
            assets_root=Path(assets_root).expanduser(),
            device=os.environ.get("MF_DEVICE", "cuda:0"),
            weights=weights,
            release_codecs_on_task_switch=release,
            precision=os.environ.get("MF_PRECISION", "bf16"),
            codec_device=os.environ.get("MF_CODEC_DEVICE") or None,
            attention_backend=os.environ.get("MF_ATTENTION_BACKEND", "flex"),
            allow_low_vram=os.environ.get("MF_ALLOW_LOW_VRAM") == "1",
        )


class MFBackend:
    """Load one MF pipeline and serialize all GPU work through it."""

    is_mock = False

    def __init__(self, options: RuntimeOptions) -> None:
        self.options = options
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._lock = threading.RLock()
        self._active_task: str | None = None
        self._status: dict[str, Any] = {
            "mode": "real",
            "loaded": False,
            "device": options.device,
            "checkpoint": str(options.checkpoint),
        }
        self.defaults: dict[str, Any] = {"steps": 64, "cfg": 2.0, "method": "sde"}

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self._pipeline is not None:
                return self.status()

            checkpoint = self.options.checkpoint.resolve()
            assets_root = self.options.assets_root.resolve()
            model_file = checkpoint / "model.safetensors"
            decoder_file = assets_root / "scale_rae_decoder" / "model.pt"
            if not (checkpoint / "config.json").is_file():
                raise BackendError(f"MF checkpoint config is missing: {checkpoint / 'config.json'}")
            if not model_file.is_file() or model_file.stat().st_size < 3_000_000_000:
                raise BackendError(f"MF checkpoint weights are missing or incomplete: {model_file}")
            if not decoder_file.is_file() or decoder_file.stat().st_size < 1_000_000_000:
                raise BackendError(
                    f"Scale RAE decoder is missing or incomplete. Expected {decoder_file}"
                )

            os.environ["MF_ASSETS_ROOT"] = str(assets_root)
            os.environ["MF_ATTENTION_BACKEND"] = self.options.attention_backend
            try:
                import torch
                from mf.inference.pipeline import MFPipeline
            except ImportError as error:
                raise BackendError(
                    "MF or PyTorch is not installed. Install the repository environment "
                    "and the Gradio requirements first."
                ) from error

            if not self.options.device.startswith("cuda"):
                raise BackendError("The production demo requires an NVIDIA CUDA device.")
            if not torch.cuda.is_available():
                raise BackendError(
                    "CUDA is unavailable to PyTorch. Check the driver and WSL setup."
                )

            index = torch.device(self.options.device).index or 0
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            vram_gib = properties.total_memory / (1024**3)
            with torch.cuda.device(index):
                bf16_supported = torch.cuda.is_bf16_supported()
            if self.options.precision == "bf16" and (capability[0] < 8 or not bf16_supported):
                raise BackendError(
                    f"{properties.name} does not provide the native BF16 support required "
                    "for this demo. Use an Ampere-or-newer GPU."
                )
            if vram_gib < MIN_SUPPORTED_VRAM_GIB and not self.options.allow_low_vram:
                raise BackendError(
                    f"Only {vram_gib:.1f} GiB VRAM is visible. A single GPU with at least "
                    f"{MIN_SUPPORTED_VRAM_GIB:.0f} GiB is required even for an experimental "
                    "run. "
                    "VRAM from multiple cards is not combined automatically."
                )
            if self.options.codec_device is not None:
                codec_device = torch.device(self.options.codec_device)
                if codec_device.type == "cuda":
                    torch.cuda.get_device_properties(codec_device)
            vram_warning = (
                f"Only {vram_gib:.1f} GiB VRAM is visible; 20+ GiB is recommended. "
                "Close other GPU applications and expect possible out-of-memory errors."
                if vram_gib < MIN_RECOMMENDED_VRAM_GIB
                else None
            )

            started = time.perf_counter()
            try:
                pipeline = MFPipeline.from_checkpoint(
                    checkpoint,
                    device=self.options.device,
                    weights=self.options.weights,
                    inference_dtype=(torch.float16 if self.options.precision == "fp16" else None),
                    codec_device=self.options.codec_device,
                )
            except Exception as error:
                raise BackendError(f"MF-1 failed to load: {error}") from error

            sampler = pipeline.default_sampler_config
            self._pipeline = pipeline
            self._torch = torch
            self.defaults = {
                "steps": int(sampler.num_inference_steps),
                "cfg": float(sampler.cfg_scale),
                "method": str(sampler.method),
            }
            self._status = {
                "mode": "real",
                "loaded": True,
                "model": str(pipeline.config.model.name),
                "device": properties.name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "vram_gib": round(vram_gib, 1),
                "weights": self.options.weights,
                "checkpoint": str(checkpoint),
                "load_seconds": round(time.perf_counter() - started, 2),
                "release_codecs_on_task_switch": (self.options.release_codecs_on_task_switch),
                "precision": self.options.precision,
                "codec_precision": "fp32" if self.options.precision == "fp16" else "bf16",
                "codec_device": self.options.codec_device or self.options.device,
                "attention_backend": self.options.attention_backend,
                "experimental": self.options.precision != "bf16" or self.options.allow_low_vram,
            }
            if vram_warning is not None:
                self._status["vram_warning"] = vram_warning
            return self.status()

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            self.load()
        return self._pipeline

    def _prepare_task(self, name: str) -> Any:
        pipeline = self._require_pipeline()
        if (
            self.options.release_codecs_on_task_switch
            and self._active_task is not None
            and self._active_task != name
        ):
            pipeline.bundle.close()
            gc.collect()
            with self._torch.cuda.device(self.options.device):
                self._torch.cuda.empty_cache()
            if self.options.codec_device and self.options.codec_device.startswith("cuda"):
                with self._torch.cuda.device(self.options.codec_device):
                    self._torch.cuda.empty_cache()
        self._active_task = name
        self._torch.cuda.reset_peak_memory_stats(self.options.device)
        if self.options.codec_device and self.options.codec_device.startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats(self.options.codec_device)
        return pipeline

    def _metadata(self, *, task: str, seed: int, started: float, **settings: Any) -> dict[str, Any]:
        self._torch.cuda.synchronize(self.options.device)
        if self.options.codec_device and self.options.codec_device.startswith("cuda"):
            self._torch.cuda.synchronize(self.options.codec_device)
        peak = self._torch.cuda.max_memory_allocated(self.options.device) / (1024**3)
        devices = {self.options.device}
        if self.options.codec_device and self.options.codec_device.startswith("cuda"):
            devices.add(self.options.codec_device)
        return {
            "task": task,
            "seed": seed,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_allocated_vram_gib": round(peak, 2),
            "peak_allocated_vram_by_device_gib": {
                device: round(self._torch.cuda.max_memory_allocated(device) / (1024**3), 2)
                for device in sorted(devices)
            },
            "precision": self.options.precision,
            "attention_backend": self.options.attention_backend,
            **settings,
        }

    def generate_image(
        self,
        prompt: str,
        steps: int | float,
        cfg: int | float,
        method: str,
        seed: int | float | None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        prompt = require_text(prompt, "Prompt")
        steps = _integer(steps, "Inference steps", minimum=1, maximum=128)
        cfg = _number(cfg, "CFG scale", minimum=0.0, maximum=12.0)
        seed = normalize_seed(seed)
        if method not in {"ode", "sde"}:
            raise BackendError("Sampler method must be 'ode' or 'sde'.")

        with self._lock:
            pipeline = self._prepare_task("text-to-image")
            from mf.inference.protocol import GenerationConfig

            started = time.perf_counter()
            config = GenerationConfig(
                num_inference_steps=steps,
                cfg_scale=cfg,
                method=method,
                seed=seed,
            )
            tensor = pipeline.generate_image((prompt,), config=config)[0]
            image = tensor_to_pil(tensor)
            return image, self._metadata(
                task="text-to-image",
                seed=seed,
                started=started,
                prompt=prompt,
                steps=steps,
                cfg_scale=cfg,
                method=method,
            )

    def generate_unconditional(
        self,
        steps: int | float,
        method: str,
        seed: int | float | None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        steps = _integer(steps, "Inference steps", minimum=1, maximum=128)
        seed = normalize_seed(seed)
        if method not in {"ode", "sde"}:
            raise BackendError("Sampler method must be 'ode' or 'sde'.")

        with self._lock:
            pipeline = self._prepare_task("unconditional-image")
            from mf.inference.protocol import GenerationConfig

            started = time.perf_counter()
            config = GenerationConfig(num_inference_steps=steps, method=method, seed=seed)
            tensor = pipeline.generate_image_unconditional(1, config=config)[0]
            image = tensor_to_pil(tensor)
            return image, self._metadata(
                task="unconditional-image",
                seed=seed,
                started=started,
                steps=steps,
                method=method,
            )

    def caption(
        self,
        image: Image.Image | None,
        question: str,
        target_length: int | float,
        steps: int | float,
        cfg: int | float,
        seed: int | float | None,
    ) -> tuple[str, dict[str, Any]]:
        if image is None:
            raise BackendError("Upload an image first.")
        question = require_text(question, "Question or instruction")
        target_length = _integer(target_length, "Maximum answer length", minimum=8, maximum=512)
        steps = _integer(steps, "Inference steps", minimum=1, maximum=64)
        cfg = _number(cfg, "CFG scale", minimum=0.0, maximum=12.0)
        seed = normalize_seed(seed)

        with self._lock:
            pipeline = self._prepare_task("image-to-text")
            from mf.data.images import preprocess_image
            from mf.inference.protocol import GenerationConfig

            vision = pipeline.config.codecs.vision
            tensor = preprocess_image(
                image,
                resolution=vision.encoder_input_resolution,
                policy=getattr(
                    vision,
                    "image_preprocessing",
                    "legacy_center_crop_bicubic_v1",
                ),
            )
            started = time.perf_counter()
            config = GenerationConfig(
                num_inference_steps=steps,
                cfg_scale=cfg,
                seed=seed,
            )
            text = pipeline.caption(
                (tensor,),
                prompt=question,
                target_length=target_length,
                config=config,
            )[0]
            return text, self._metadata(
                task="image-to-text",
                seed=seed,
                started=started,
                question=question,
                target_length=target_length,
                steps=steps,
                cfg_scale=cfg,
            )

    def continue_text(
        self,
        prompt: str,
        target_length: int | float,
        steps: int | float,
        cfg: int | float,
        seed: int | float | None,
        stop: str | None = "",
    ) -> tuple[str, dict[str, Any]]:
        prompt = require_text(prompt, "Text prefix")
        target_length = _integer(
            target_length, "Maximum continuation length", minimum=8, maximum=512
        )
        steps = _integer(steps, "Inference steps", minimum=1, maximum=64)
        cfg = _number(cfg, "CFG scale", minimum=0.0, maximum=12.0)
        seed = normalize_seed(seed)
        stops = tuple(value.strip() for value in (stop or "").splitlines() if value.strip())

        with self._lock:
            pipeline = self._prepare_task("text-continuation")
            from mf.inference.protocol import GenerationConfig

            started = time.perf_counter()
            config = GenerationConfig(
                num_inference_steps=steps,
                cfg_scale=cfg,
                seed=seed,
            )
            text = pipeline.complete_text(
                (prompt,),
                target_length=target_length,
                stop=stops,
                config=config,
            )[0]
            return text, self._metadata(
                task="text-continuation",
                seed=seed,
                started=started,
                prompt=prompt,
                target_length=target_length,
                steps=steps,
                cfg_scale=cfg,
                stop=list(stops),
            )

    def close(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.close()
            self._pipeline = None
            self._active_task = None
            self._status["loaded"] = False
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_available():
                with self._torch.cuda.device(self.options.device):
                    self._torch.cuda.empty_cache()
                if self.options.codec_device and self.options.codec_device.startswith("cuda"):
                    with self._torch.cuda.device(self.options.codec_device):
                        self._torch.cuda.empty_cache()


class MockBackend:
    """Deterministic lightweight backend for UI development and automated tests."""

    is_mock = True
    defaults = {"steps": 16, "cfg": 2.0, "method": "sde"}

    def load(self) -> dict[str, Any]:
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "mode": "mock",
            "loaded": True,
            "model": "MF-1 mock backend",
            "device": "No GPU used",
            "warning": "Outputs are simulated and are not MF-1 results.",
        }

    @staticmethod
    def _image(seed: int, label: str) -> Image.Image:
        rng = random.Random(seed)
        top = tuple(rng.randint(20, 110) for _ in range(3))
        bottom = tuple(rng.randint(130, 245) for _ in range(3))
        image = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(image)
        for y in range(512):
            amount = y / 511
            color = tuple(int(a + (b - a) * amount) for a, b in zip(top, bottom))
            draw.line(((0, y), (512, y)), fill=color)
        for _ in range(18):
            x, y = rng.randrange(512), rng.randrange(512)
            radius = rng.randrange(8, 70)
            color = tuple(rng.randint(140, 255) for _ in range(3))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        image = image.filter(ImageFilter.GaussianBlur(2.0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 454, 494, 494), fill=(10, 15, 28))
        draw.text((30, 466), f"MOCK · {label[:48]}", fill=(235, 242, 255))
        return image

    @staticmethod
    def _meta(task: str, seed: int, **settings: Any) -> dict[str, Any]:
        return {
            "task": task,
            "seed": seed,
            "elapsed_seconds": 0.0,
            "mock": True,
            **settings,
        }

    def generate_image(self, prompt, steps, cfg, method, seed):
        prompt = require_text(prompt, "Prompt")
        seed = normalize_seed(seed)
        return self._image(seed, prompt), self._meta(
            "text-to-image",
            seed,
            prompt=prompt,
            steps=int(steps),
            cfg_scale=float(cfg),
            method=method,
        )

    def generate_unconditional(self, steps, method, seed):
        seed = normalize_seed(seed)
        return self._image(seed, "unconditional"), self._meta(
            "unconditional-image", seed, steps=int(steps), method=method
        )

    def caption(self, image, question, target_length, steps, cfg, seed):
        if image is None:
            raise BackendError("Upload an image first.")
        question = require_text(question, "Question or instruction")
        seed = normalize_seed(seed)
        text = (
            f"[Mock output] The uploaded image is {image.width}×{image.height} pixels. "
            f"The requested instruction was: {question}"
        )
        return text, self._meta(
            "image-to-text",
            seed,
            question=question,
            target_length=int(target_length),
            steps=int(steps),
            cfg_scale=float(cfg),
        )

    def continue_text(self, prompt, target_length, steps, cfg, seed, stop=""):
        prompt = require_text(prompt, "Text prefix")
        seed = normalize_seed(seed)
        text = f"{prompt} [mock continuation generated with seed {seed}]"
        return text, self._meta(
            "text-continuation",
            seed,
            prompt=prompt,
            target_length=int(target_length),
            steps=int(steps),
            cfg_scale=float(cfg),
            stop=[value.strip() for value in stop.splitlines() if value.strip()],
        )

    def close(self) -> None:
        return None
