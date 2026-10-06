"""Record observed GPU memory, utilization, temperature, and power."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


class GPUMonitor:
    def __init__(self, path: Path, interval: float = 0.5) -> None:
        self.path = path
        self.interval = interval
        self.samples: list[list[str]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_unix",
                    "physical_index",
                    "uuid",
                    "name",
                    "memory_used_mib",
                    "memory_total_mib",
                    "utilization_percent",
                    "temperature_c",
                    "power_watts",
                ]
            )
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=3,
                    )
                    timestamp = str(time.time())
                    for row in csv.reader(result.stdout.splitlines()):
                        sample = [timestamp, *(value.strip() for value in row)]
                        self.samples.append(sample)
                        writer.writerow(sample)
                    stream.flush()
                except (OSError, subprocess.SubprocessError) as error:
                    self.errors.append(str(error))
                self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        devices = {}
        for sample in self.samples:
            _, index, uuid, name, used, total, utilization, _, _ = sample
            device = devices.setdefault(
                index,
                {
                    "uuid": uuid,
                    "name": name,
                    "total_mib": float(total),
                    "peak_used_mib": 0.0,
                    "peak_utilization_percent": 0.0,
                },
            )
            device["peak_used_mib"] = max(device["peak_used_mib"], float(used))
            device["peak_utilization_percent"] = max(
                device["peak_utilization_percent"], float(utilization)
            )
        return {
            "csv": str(self.path.resolve()),
            "interval_seconds": self.interval,
            "samples": len(self.samples),
            "devices": devices,
            "errors": self.errors,
            "scope": "whole-device memory, including other processes; sampled peaks",
        }
