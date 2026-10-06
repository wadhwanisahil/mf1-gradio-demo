# MF-1 Gradio research demo

A review-ready Gradio interface for the released **Multimodal Flow (MF-1)**
checkpoint. The application uses the repository's public `MFPipeline` API and
supports the three principal inference paths:

- text-to-image and unconditional image generation;
- image captioning and visual question answering;
- text continuation.

Every result includes its seed, generation settings, elapsed time, and peak
allocated GPU memory. The real backend keeps one checkpoint loaded, accepts one
GPU request at a time, and releases task-specific codecs when the user switches
modalities to make 24 GB cards more practical.

The integration also carries two narrowly scoped released-checkpoint
compatibility fixes in the repository source: the Hugging Face loader ignores
the explicitly known removed legacy guidance parameters, and inference configs
default `gradient_checkpointing` to `false`. These changes should remain separate
and visible in review rather than being hidden in UI monkey-patches.

> Mock mode is provided for UI development and automated tests. It is visibly
> labelled and must never be presented as MF-1 output.

## Hardware

The professor-ready target is one NVIDIA Ampere-or-newer GPU with at least
20 GiB of visible VRAM. An RTX 3090 (24 GB), A10/A10G (24 GB), L4 (24 GB),
A100, or newer card is appropriate.

- One RTX 3090 should be sufficient, subject to the final smoke test.
- A 12 GB RTX 3060 is allowed as an experimental run, but may run out of memory
  and should not replace validation on the 3090.
- An 8 GB card fails the default configuration's hardware checks. The opt-in
  compatibility configuration below uses FP16 and can place codecs on a second
  GPU; it must be validated separately from the released BF16 configuration.
- Two 8 GB cards do **not** become a 16 GB device. MF inference is single-device
  and this demo does not shard the backbone. Codec placement on another GPU is
  explicit and does not combine the cards' memory into one device.
- Close other GPU-heavy applications before loading the model. A display attached
  to the same GPU also consumes VRAM.

Native Windows is not the reference environment because MF depends on a
Linux-oriented CUDA/Triton/FlashAttention stack. On a Windows 10 GPU workstation,
use **WSL2 with Ubuntu 22.04**.

## Windows 10 + WSL2 preparation

1. Install the latest NVIDIA Windows driver that supports CUDA in WSL.
2. In an Administrator PowerShell window:

   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --update
   ```

3. Restart Windows if requested, open Ubuntu, and verify:

   ```bash
   nvidia-smi
   ```

Do not install a separate Linux display driver inside WSL. The Windows NVIDIA
driver exposes the GPU to WSL. Keep the repository under the Linux filesystem
(for example `~/Multimodal-Flow`) rather than `/mnt/c` for faster package builds
and model loading.

References: [Microsoft CUDA in WSL](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)
and the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Install

From the submitted repository checkout (which contains this demo and the two
checkpoint-compatibility fixes), run under Ubuntu/Linux:

```bash
cd Multimodal-Flow
conda env create -f environment.yaml
conda activate multimodal-flow
pip install -e .
pip install -r gradio_demo/requirements.txt
```

The official environment uses Python 3.11 and pins the MF dependencies. Do not
replace it with the unrelated packages from a system Python installation.
The demo pins Gradio 6.17.3 and preserves the official Hugging Face Hub 0.36.2
version. Gradio 6.29.1 requires Hub 1.x or newer, which is incompatible
with the official Transformers 4.57.1 dependency. After installing, verify the
combined environment with `python -m pip check`.

## Compatibility configuration for two RTX 2080 SUPER GPUs

The [6 October 2026 experiment report](../docs/experiments/2026-10-06-turing/REPORT.md)
includes real outputs for all three task paths, 14 controlled experiments,
GPU measurements, and live API/browser validation on this configuration.

This optional configuration retains the official checkpoint, architecture,
samplers, and chunk visibility rules. The backbone uses FP16; encoders and the
image decoder remain FP32. PyTorch SDPA evaluates the exact MF chunk mask instead
of compiled FlexAttention. Encoders and the image decoder can run on a second
GPU, returning their outputs to the backbone's device.

It is an experimental numerical configuration, not proof of parity with native
BF16. Never enable it merely to bypass checks without running the real tests.
The default BF16 behavior and its hardware requirements remain available.

Create the compatibility environment from the repository root:

```bash
conda env create -f gradio_demo/environment-compat.yaml
conda activate multimodal-flow-compat
python -m pip check
```

This retains the official MF version pins and omits FlashAttention, which is
optional for the SDPA path and whose standard CUDA implementation does not
support the RTX 2080 SUPER. Select both physical GPUs before starting Python:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MF_DEVICE=cuda:0
export MF_CODEC_DEVICE=cuda:1
export MF_PRECISION=fp16
export MF_ATTENTION_BACKEND=sdpa
export MF_ALLOW_LOW_VRAM=1
```

Download assets and set `MF_CHECKPOINT` and `MF_ASSETS_ROOT` as described below.
Run preflight, smoke testing, and the controlled experiments:

```bash
python gradio_demo/scripts/preflight.py
python gradio_demo/scripts/smoke_test.py \
  --checkpoint "$MF_CHECKPOINT" --assets-root "$MF_ASSETS_ROOT"
python gradio_demo/scripts/experiments.py
```

`experiments.py` runs 14 real requests: image generation with 16/32/64 steps
using ODE and SDE, a second seed and a same-seed repeat, and caption/text
continuation with 8/16/32 steps. It writes every output and a report incrementally
to `outputs/mf1-experiments/`. This is a controlled demonstration, not a dataset
benchmark. Reports include experimental configuration details, per-device
allocated memory peaks, and sampled whole-device measurements from `nvidia-smi`.

To test all four real endpoints, start the application in a separate terminal
with the same environment, then run:

```bash
python gradio_demo/scripts/test_endpoints.py
```

The endpoint checker rejects mock or unloaded servers. `MF_CODEC_DEVICE=cpu` is
also available for explicit codec offloading, but timings will differ. The
backbone continues to require CUDA.

On a machine with insufficient disk space, `HF_HOME`, model/asset destinations,
and temporary package storage can point to a sufficiently large RAM-backed
filesystem. **RAM-backed assets and packages disappear on reboot.** Preserve
the small output artifacts and report on persistent storage; plan disk capacity
for a durable installation.

## Download only the required assets

From the repository root:

```bash
python gradio_demo/scripts/download_assets.py \
  --model-root "$PWD/MF_weights" \
  --assets-root "$PWD/assets"
```

This downloads:

- `hustvl/Multimodal-Flow`: SFT checkpoint, text decoder, and vision statistics;
- `nyu-visionx/siglip2_decoder`: Scale RAE image decoder.

T5-small and SigLIP2 are resolved through Hugging Face on first use. Allow
additional disk and download time for their caches.

Configure the application:

```bash
export MF_CHECKPOINT="$PWD/MF_weights/MF/sft"
export MF_ASSETS_ROOT="$PWD/assets"
export MF_DEVICE="cuda:0"
export MF_WEIGHTS="ema"
```

To select a specific physical card before starting Python:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The selected card is then exposed to the application as `cuda:0`.

## Preflight validation

Run this before loading the model:

```bash
python gradio_demo/scripts/preflight.py
```

It checks package availability, CUDA, native BF16, visible VRAM, disk space,
checkpoint sizes, shared decoder/statistics, and Scale RAE assets. Any `FAIL`
must be resolved before claiming that the demo works.

## Mandatory real-model smoke test

Run the three inference paths and preserve their outputs:

```bash
python gradio_demo/scripts/smoke_test.py \
  --checkpoint "$MF_CHECKPOINT" \
  --assets-root "$MF_ASSETS_ROOT" \
  --output-dir outputs/mf1-smoke
```

Successful evidence includes:

```text
outputs/mf1-smoke/
  image_understanding.txt
  report.json
  text_continuation.txt
  text_to_image.png
```

`report.json` records settings, timings, device information, and peak allocated
VRAM. Keep these files for the project report and demonstration video. A mock
run does not replace this smoke test.

## Launch

Production mode preloads and validates MF-1 before opening the web server:

```bash
python gradio_demo/app.py
```

Open <http://127.0.0.1:7860>. Windows normally forwards this WSL address to the
host browser.

Useful options:

```bash
python gradio_demo/app.py --mock
python gradio_demo/app.py --lazy
python gradio_demo/app.py --share
python gradio_demo/app.py --host 0.0.0.0 --port 7860
```

- `--mock` previews and tests the UI without MF, weights, or CUDA.
- `--lazy` defers model loading until the first request.
- `--share` requests a temporary public Gradio URL.
- `--host 0.0.0.0` exposes the service to the network; use firewall rules and
  authentication on any untrusted network.

The following environment variables are also supported:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MF_CHECKPOINT` | required | Released `MF/sft` directory |
| `MF_ASSETS_ROOT` | required | Parent of `scale_rae_decoder/` |
| `MF_DEVICE` | `cuda:0` | Single CUDA device used for inference |
| `MF_PRECISION` | `bf16` | Backbone inference precision; `fp16` is experimental |
| `MF_CODEC_DEVICE` | backbone device | Optional second GPU or CPU for frozen codecs |
| `MF_ATTENTION_BACKEND` | `flex` | `flex` or exact-mask PyTorch `sdpa` |
| `MF_ALLOW_LOW_VRAM` | `0` | Explicit experimental override of the 10 GiB check |
| `MF_WEIGHTS` | `ema` | `ema` or `raw` checkpoint weights |
| `MF_RELEASE_CODECS_ON_TASK_SWITCH` | `1` | Release lazy modality codecs when switching tasks |
| `MF_MOCK` | `0` | Set to `1` for mock mode |
| `MF_LOG_LEVEL` | `INFO` | Python logging level |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address |
| `GRADIO_SERVER_PORT` | `7860` | Bind port |
| `GRADIO_SHARE` | `false` | Request a temporary public URL |

## Tests

The tests do not require MF weights or a GPU:

```bash
PYTHONPATH=gradio_demo python -m unittest discover -s gradio_demo/tests -v
```

Static checks:

```bash
ruff check gradio_demo
ruff format --check gradio_demo
```

## Submission checklist

Before sending the work to a reviewer:

- [ ] Preflight reports no failures.
- [ ] The real smoke test completes all three tasks.
- [ ] `report.json` identifies the actual GPU and contains no error.
- [ ] Generated image and text outputs have been visually reviewed.
- [ ] The application starts from a clean shell using only documented commands.
- [ ] Tests and static checks pass.
- [ ] A short screen recording shows the real application and all three tasks.
- [ ] The commit/PR contains source and documentation, never downloaded weights.

## Troubleshooting

### Less than 20 GiB VRAM is reported

Between 10 and 20 GiB, preflight reports a warning and the application permits an
experimental run. Below 10 GiB, it stops instead of failing later with a CUDA
out-of-memory error. Use one 24 GB or larger GPU for final validation. Multiple
smaller GPUs are not combined.

### The model loads but switching tasks runs out of memory

Keep `MF_RELEASE_CODECS_ON_TASK_SWITCH=1`, close other GPU applications, and
restart the process. Task-specific encoders/decoders are unloaded when the task
changes, while the MF backbone remains loaded.

### The checkpoint is reported as incomplete

Delete only the incomplete Hugging Face download fragment and rerun
`download_assets.py`. The released SFT `model.safetensors` is approximately
3.28 GB, the shared text decoder approximately 71.6 MB, and the Scale RAE
decoder approximately 1.66 GB.

### WSL cannot see CUDA

Update WSL and the NVIDIA **Windows** driver, restart Windows, and run
`nvidia-smi` inside Ubuntu before installing MF. Do not continue until it lists
the intended GPU.
