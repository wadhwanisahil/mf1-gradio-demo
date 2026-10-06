"""Exercise a live, preloaded real MF-1 application and preserve evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from gradio_client import Client, handle_file
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from mf_demo.monitor import GPUMonitor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mf1-endpoints"))
    parser.add_argument(
        "--input-image", type=Path, default=DEMO_DIR.parent / "docs/assets/astronaut.png"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"url": args.url, "started_at_unix": time.time(), "endpoints": []}
    monitor = GPUMonitor(output / "gpu.csv")
    monitor.start()
    try:
        config = httpx.get(args.url.rstrip("/") + "/config", timeout=30).json()
        runtime = next(
            component["props"]["value"]
            for component in config["components"]
            if component.get("props", {}).get("label") == "MF-1 runtime"
        )
        if runtime.get("mode") != "real" or not runtime.get("loaded"):
            raise RuntimeError("endpoint validation requires a preloaded real MF backend")
        report["runtime"] = runtime
        client = Client(args.url, verbose=False)
        requests = [
            (
                "generate_image",
                "text-to-image",
                [
                    "A quiet observatory above the clouds, cinematic moonlight.",
                    32,
                    2.0,
                    "sde",
                    42,
                    False,
                ],
            ),
            ("generate_unconditional", "unconditional-image", [32, "sde", 42, False]),
            (
                "analyze_image",
                "image-to-text",
                [
                    handle_file(str(args.input_image)),
                    "Describe this image in detail.",
                    64,
                    16,
                    2.0,
                    42,
                    False,
                ],
            ),
            (
                "continue_text",
                "text-continuation",
                [
                    "Continuous multimodal representations make it possible to",
                    64,
                    16,
                    2.0,
                    42,
                    False,
                    None,
                ],
            ),
        ]
        for name, task, inputs in requests:
            print(f"Calling /{name}...", flush=True)
            entry = {"endpoint": "/" + name, "task": task}
            try:
                result, metadata, seed, randomized = client.predict(*inputs, api_name="/" + name)
                if metadata.get("mock") or metadata["task"] != task or seed != 42 or randomized:
                    raise RuntimeError("invalid real-inference response metadata")
                if metadata.get("peak_allocated_vram_gib", 0) <= 0:
                    raise RuntimeError("response lacks GPU inference evidence")
                if task.endswith("image"):
                    path = output / (name + ".png")
                    with Image.open(result) as image:
                        image.convert("RGB").save(path, format="PNG")
                else:
                    if not result.strip():
                        raise RuntimeError("empty model text")
                    path = output / (name + ".txt")
                    path.write_text(result, encoding="utf-8")
                    entry["text"] = result
                entry.update(status="passed", metadata=metadata, artifact=str(path))
                print(f"Passed /{name}: {metadata['elapsed_seconds']} seconds", flush=True)
            except Exception as error:  # noqa: BLE001 - preserve individual endpoint failures
                entry.update(status="failed", error=repr(error))
                print(f"Failed /{name}: {error}", flush=True)
            report["endpoints"].append(entry)
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        report["error"] = repr(error)
    finally:
        monitor.stop()
        report["gpu_monitor"] = monitor.summary()
        report["completed_at_unix"] = time.time()
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(item["status"] == "passed" for item in report["endpoints"])
    print(f"{passed}/4 real endpoints passed. Evidence: {output}", flush=True)
    return 0 if passed == 4 and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
