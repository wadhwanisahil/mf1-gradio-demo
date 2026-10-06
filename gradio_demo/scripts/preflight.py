"""Validate an MF-1 demo host before loading several gigabytes of weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Check:
    level: str
    name: str
    detail: str


def _path_from_arg(value: Path | None, environment_name: str) -> Path | None:
    if value is not None:
        return value.expanduser().resolve()
    environment_value = os.environ.get(environment_name)
    return Path(environment_value).expanduser().resolve() if environment_value else None


def _file_check(name: str, path: Path, minimum_bytes: int) -> Check:
    if not path.is_file():
        return Check("FAIL", name, f"missing: {path}")
    size = path.stat().st_size
    if size < minimum_bytes:
        return Check(
            "FAIL",
            name,
            f"incomplete: {path} ({size / 1_000_000:.1f} MB)",
        )
    return Check("PASS", name, f"{path} ({size / 1_000_000:.1f} MB)")


def inspect_host(
    checkpoint: Path | None,
    assets_root: Path | None,
    *,
    device: str = "cuda:0",
    precision: str = "bf16",
    allow_low_vram: bool = False,
    codec_device: str | None = None,
) -> list[Check]:
    checks = [
        Check("INFO", "platform", platform.platform()),
        Check("INFO", "python", sys.version.split()[0]),
        Check(
            "INFO",
            "free disk",
            f"{shutil.disk_usage(Path.cwd()).free / (1024**3):.1f} GiB",
        ),
    ]

    for package in ("torch", "torchvision", "transformers", "gradio", "multimodal-flow"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check("FAIL", f"package {package}", "not installed"))
        else:
            checks.append(Check("PASS", f"package {package}", version))

    try:
        import torch
    except Exception as error:  # noqa: BLE001 - diagnostics must report binary import errors
        checks.append(Check("FAIL", "PyTorch import", repr(error)))
    else:
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            checks.append(
                Check(
                    "FAIL",
                    "PyTorch installation",
                    "the imported torch package has no CUDA module; reinstall PyTorch",
                )
            )
        elif not cuda.is_available():
            checks.append(Check("FAIL", "CUDA", "not available to PyTorch"))
        else:
            selected = torch.device(device)
            properties = cuda.get_device_properties(selected)
            capability = cuda.get_device_capability(selected)
            vram = properties.total_memory / (1024**3)
            checks.append(
                Check(
                    "PASS",
                    "CUDA device",
                    f"{properties.name}; {vram:.1f} GiB; capability {capability[0]}.{capability[1]}",
                )
            )
            with cuda.device(selected):
                bf16_supported = cuda.is_bf16_supported()
            if precision == "fp16":
                checks.append(Check("WARN", "precision", "experimental FP16; encoders remain FP32"))
            elif capability[0] < 8 or not bf16_supported:
                checks.append(Check("FAIL", "native BF16", "not supported"))
            else:
                checks.append(Check("PASS", "native BF16", "supported"))
            if vram < 10:
                checks.append(
                    Check(
                        "WARN" if allow_low_vram else "FAIL",
                        "minimum VRAM",
                        f"{vram:.1f} GiB visible; low-VRAM opt-in: {allow_low_vram}",
                    )
                )
            elif vram < 20:
                checks.append(
                    Check(
                        "WARN",
                        "recommended VRAM",
                        f"{vram:.1f} GiB visible; 20+ GiB is recommended for final validation",
                    )
                )
            else:
                checks.append(Check("PASS", "recommended VRAM", f"{vram:.1f} GiB"))
            if codec_device is not None:
                target = torch.device(codec_device)
                if target.type == "cuda":
                    codec_properties = cuda.get_device_properties(target)
                    checks.append(
                        Check("PASS", "codec device", f"{target}: {codec_properties.name}")
                    )
                else:
                    checks.append(Check("INFO", "codec device", str(target)))

    if checkpoint is None:
        checks.append(Check("FAIL", "MF_CHECKPOINT", "not provided"))
    else:
        checks.append(_file_check("checkpoint config", checkpoint / "config.json", 1_000))
        checks.append(
            _file_check(
                "MF-1 SFT weights",
                checkpoint / "model.safetensors",
                3_000_000_000,
            )
        )
        config_path = checkpoint / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                for label, key, threshold in (
                    ("text decoder", "text_decoder_path", 60_000_000),
                    ("vision statistics", "vision_statistics_path", 2_000_000),
                ):
                    source = config.get(key)
                    if not source:
                        checks.append(Check("FAIL", label, f"{key} is absent from config"))
                    else:
                        checks.append(
                            _file_check(label, (checkpoint / source).resolve(), threshold)
                        )
            except (OSError, ValueError) as error:
                checks.append(Check("FAIL", "checkpoint config parse", repr(error)))

    if assets_root is None:
        checks.append(Check("FAIL", "MF_ASSETS_ROOT", "not provided"))
    else:
        decoder = assets_root / "scale_rae_decoder"
        checks.append(_file_check("Scale RAE config", decoder / "config.json", 500))
        checks.append(_file_check("Scale RAE weights", decoder / "model.pt", 1_000_000_000))

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--device", default=os.environ.get("MF_DEVICE", "cuda:0"))
    parser.add_argument(
        "--precision", choices=("bf16", "fp16"), default=os.environ.get("MF_PRECISION", "bf16")
    )
    parser.add_argument("--codec-device", default=os.environ.get("MF_CODEC_DEVICE"))
    parser.add_argument(
        "--allow-low-vram", action="store_true", default=os.environ.get("MF_ALLOW_LOW_VRAM") == "1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = _path_from_arg(args.checkpoint, "MF_CHECKPOINT")
    assets_root = _path_from_arg(args.assets_root, "MF_ASSETS_ROOT")
    checks = inspect_host(
        checkpoint,
        assets_root,
        device=args.device,
        precision=args.precision,
        allow_low_vram=args.allow_low_vram,
        codec_device=args.codec_device,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            print(f"[{check.level:4}] {check.name}: {check.detail}")
        passed = sum(check.level == "PASS" for check in checks)
        failed = sum(check.level == "FAIL" for check in checks)
        print(f"\nSummary: {passed} passed, {failed} failed")
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
