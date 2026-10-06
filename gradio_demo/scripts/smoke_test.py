"""Run and preserve evidence for MF-1's three principal inference paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from mf_demo.backend import MFBackend, RuntimeOptions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mf1-smoke"))
    parser.add_argument(
        "--input-image",
        type=Path,
        default=REPO_ROOT / "docs" / "assets" / "astronaut.png",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--image-steps", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    options = RuntimeOptions(
        checkpoint=args.checkpoint,
        assets_root=args.assets_root,
        device=args.device,
        weights=args.weights,
    )
    backend = MFBackend(options)
    evidence: dict[str, object] = {
        "started_at_unix": time.time(),
        "input_image": str(args.input_image.resolve()),
    }
    try:
        print("Loading MF-1...", flush=True)
        evidence["runtime"] = backend.load()

        print("1/3 Text continuation", flush=True)
        text, text_meta = backend.continue_text(
            "Continuous multimodal representations make it possible to",
            128,
            16,
            2.0,
            42,
        )
        (output_dir / "text_continuation.txt").write_text(text, encoding="utf-8")
        evidence["text_continuation"] = {"output": text, "metadata": text_meta}

        print("2/3 Image understanding", flush=True)
        with Image.open(args.input_image) as source:
            caption, caption_meta = backend.caption(
                source.convert("RGB"),
                "Describe this image in detail.",
                128,
                16,
                2.0,
                42,
            )
        (output_dir / "image_understanding.txt").write_text(caption, encoding="utf-8")
        evidence["image_understanding"] = {
            "output": caption,
            "metadata": caption_meta,
        }

        print("3/3 Text-to-image", flush=True)
        image, image_meta = backend.generate_image(
            "A quiet observatory above the clouds, cinematic moonlight.",
            args.image_steps,
            2.0,
            "sde",
            42,
        )
        image.save(output_dir / "text_to_image.png")
        evidence["text_to_image"] = {"metadata": image_meta}
        evidence["completed_at_unix"] = time.time()
        (output_dir / "report.json").write_text(
            json.dumps(evidence, indent=2),
            encoding="utf-8",
        )
        print(f"Smoke test passed. Evidence saved to {output_dir}")
        return 0
    except Exception as error:  # noqa: BLE001 - persist the exact integration failure
        evidence["error"] = repr(error)
        evidence["failed_at_unix"] = time.time()
        (output_dir / "report.json").write_text(
            json.dumps(evidence, indent=2),
            encoding="utf-8",
        )
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
