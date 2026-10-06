"""Run a controlled, real-model comparison of steps, sampler, and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from mf_demo.backend import MFBackend, RuntimeOptions  # noqa: E402
from mf_demo.monitor import GPUMonitor  # noqa: E402

IMAGE_PROMPT = "A quiet observatory above the clouds, cinematic moonlight."
TEXT_PROMPT = "Continuous multimodal representations make it possible to"
QUESTION = "Describe this image in detail."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mf1-experiments"))
    parser.add_argument(
        "--input-image", type=Path, default=DEMO_DIR.parent / "docs/assets/astronaut.png"
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = MFBackend(RuntimeOptions.from_env())
    monitor = GPUMonitor(output / "gpu.csv")
    report = {
        "started_at_unix": time.time(),
        "experiments": [],
        "design": {
            "image_prompt": IMAGE_PROMPT,
            "text_prompt": TEXT_PROMPT,
            "caption_question": QUESTION,
            "caption_input": str(args.input_image.resolve()),
            "cfg_scale": 2.0,
            "baseline_seed": 42,
            "scope": "controlled demonstration; not a quantitative benchmark or BF16 parity study",
        },
    }

    def save() -> None:
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def run(name, task, function, *inputs):
        print(f"Running {name}...", flush=True)
        entry = {"name": name, "task": task}
        try:
            result, metadata = function(*inputs)
            suffix = ".png" if isinstance(result, Image.Image) else ".txt"
            path = output / (name + suffix)
            if isinstance(result, Image.Image):
                result.save(path)
                entry["pixel_sha256"] = hashlib.sha256(result.tobytes()).hexdigest()
                entry["image_size"] = list(result.size)
            else:
                if not result.strip():
                    raise RuntimeError("model returned empty text")
                path.write_text(result, encoding="utf-8")
                entry["text"] = result
            entry.update(status="passed", metadata=metadata, artifact=str(path))
            print(f"Passed {name}: {metadata['elapsed_seconds']} seconds", flush=True)
        except Exception as error:  # noqa: BLE001 - retain failures and continue other tasks
            entry.update(status="failed", error=repr(error))
            print(f"Failed {name}: {error}", flush=True)
        report["experiments"].append(entry)
        save()

    monitor.start()
    try:
        report["runtime"] = backend.load()
        save()
        for steps in (16, 32, 64):
            for method in ("ode", "sde"):
                run(
                    f"image_{method}_{steps}_seed42",
                    "text-to-image",
                    backend.generate_image,
                    IMAGE_PROMPT,
                    steps,
                    2.0,
                    method,
                    42,
                )
        run(
            "image_sde_64_seed43",
            "text-to-image",
            backend.generate_image,
            IMAGE_PROMPT,
            64,
            2.0,
            "sde",
            43,
        )
        run(
            "image_sde_64_seed42_repeat",
            "text-to-image",
            backend.generate_image,
            IMAGE_PROMPT,
            64,
            2.0,
            "sde",
            42,
        )
        with Image.open(args.input_image) as source:
            image = source.convert("RGB")
        for steps in (8, 16, 32):
            run(
                f"caption_{steps}_seed42",
                "image-to-text",
                backend.caption,
                image,
                QUESTION,
                64,
                steps,
                2.0,
                42,
            )
        for steps in (8, 16, 32):
            run(
                f"text_{steps}_seed42",
                "text-continuation",
                backend.continue_text,
                TEXT_PROMPT,
                64,
                steps,
                2.0,
                42,
            )
        successful = {
            item["name"]: item for item in report["experiments"] if item["status"] == "passed"
        }
        original = successful.get("image_sde_64_seed42")
        repeat = successful.get("image_sde_64_seed42_repeat")
        changed = successful.get("image_sde_64_seed43")
        if original and repeat and changed:
            report["repeatability"] = {
                "same_seed_pixels_identical": original["pixel_sha256"] == repeat["pixel_sha256"],
                "different_seed_pixels_differ": original["pixel_sha256"] != changed["pixel_sha256"],
            }
        report["completed_at_unix"] = time.time()
    except Exception as error:  # noqa: BLE001
        report["error"] = repr(error)
    finally:
        monitor.stop()
        report["gpu_monitor"] = monitor.summary()
        save()
        backend.close()
    passed = sum(item["status"] == "passed" for item in report["experiments"])
    print(f"{passed}/14 real experiments passed. Evidence: {output}", flush=True)
    return 0 if passed == 14 and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
