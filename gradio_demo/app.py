"""Command-line entry point for the MF-1 Gradio demo."""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_demo.backend import BackendError, MFBackend, MockBackend, RuntimeOptions  # noqa: E402
from mf_demo.ui import build_demo, launch_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the MF-1 Gradio demo.")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=os.environ.get("MF_MOCK") == "1",
        help="Run a clearly labelled UI preview without MF or a GPU.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Load MF-1 on the first request instead of before the server starts.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        help="Ask Gradio to create a temporary public link.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MF_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        backend = MockBackend() if args.mock else MFBackend(RuntimeOptions.from_env())
        if not args.lazy:
            backend.load()
    except BackendError as error:
        print(f"MF-1 startup failed: {error}", file=sys.stderr)
        return 2

    atexit.register(backend.close)
    demo = build_demo(backend)
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=False,
        **launch_style(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
