"""Download only the released assets required by the MF-1 SFT demo."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("MF_weights"),
        help="Destination for the released MF model package.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("assets"),
        help="Destination whose scale_rae_decoder child will contain the image decoder.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. The required repositories are public.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface-hub is required. Install the repository environment first."
        ) from error

    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    print(f"Downloading MF-1 SFT assets to {model_root}", flush=True)
    snapshot_download(
        repo_id="hustvl/Multimodal-Flow",
        local_dir=model_root,
        allow_patterns=("MF/sft/*", "Text Decoder/*", "Vision statistics/*"),
        token=args.token,
    )

    decoder_root = assets_root / "scale_rae_decoder"
    print(f"Downloading Scale RAE decoder to {decoder_root}", flush=True)
    snapshot_download(
        repo_id="nyu-visionx/siglip2_decoder",
        local_dir=decoder_root,
        allow_patterns=("config.json", "model.pt"),
        token=args.token,
    )

    print("Downloads complete.")
    print(f"MF_CHECKPOINT={model_root / 'MF' / 'sft'}")
    print(f"MF_ASSETS_ROOT={assets_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
