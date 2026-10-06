"""Gradio UI construction for the MF-1 demo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from .backend import BackendError, MAX_SEED, choose_seed

LOGGER = logging.getLogger("mf_demo")
STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]

T2I_EXAMPLES = [
    ["A quiet observatory above the clouds, cinematic moonlight."],
    ["A red panda reading in an ancient library, detailed illustration."],
    ["A glass greenhouse on Mars at sunrise, photorealistic."],
]
TEXT_EXAMPLES = [
    ["A short language model can"],
    ["Continuous multimodal representations make it possible to"],
    ["The scientist opened the observatory door and discovered"],
]


def css() -> str:
    return (STATIC_DIR / "style.css").read_text(encoding="utf-8")


def theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="blue",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def _safe_call(function, *args):
    try:
        return function(*args)
    except BackendError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:  # noqa: BLE001 - full details remain in server logs
        LOGGER.exception("MF demo request failed")
        raise gr.Error(
            "MF-1 could not complete this request. Check the server log for details."
        ) from error


def _resolved_seed(seed: int | float | None, randomize: bool) -> int:
    return choose_seed(seed, bool(randomize))


def build_demo(backend: Any) -> gr.Blocks:
    defaults = backend.defaults
    status = backend.status()
    badge = (
        "Mock preview · no model inference"
        if backend.is_mock
        else f"Ready · {status.get('device', 'CUDA')} · {status.get('model', 'MF-1')}"
    )

    def run_image(prompt, steps, cfg, method, seed, randomize, progress=gr.Progress()):
        selected_seed = _resolved_seed(seed, randomize)
        progress(0.05, desc="Preparing MF-1 image generation")
        image, metadata = _safe_call(
            backend.generate_image,
            prompt,
            steps,
            cfg,
            method,
            selected_seed,
        )
        progress(1.0, desc="Complete")
        return image, metadata, selected_seed, False

    def run_unconditional(steps, method, seed, randomize, progress=gr.Progress()):
        selected_seed = _resolved_seed(seed, randomize)
        progress(0.05, desc="Generating an unconditional sample")
        image, metadata = _safe_call(
            backend.generate_unconditional,
            steps,
            method,
            selected_seed,
        )
        progress(1.0, desc="Complete")
        return image, metadata, selected_seed, False

    def run_caption(
        image,
        question,
        target_length,
        steps,
        cfg,
        seed,
        randomize,
        progress=gr.Progress(),
    ):
        selected_seed = _resolved_seed(seed, randomize)
        progress(0.05, desc="Encoding image and generating an answer")
        text, metadata = _safe_call(
            backend.caption,
            image,
            question,
            target_length,
            steps,
            cfg,
            selected_seed,
        )
        progress(1.0, desc="Complete")
        return text, metadata, selected_seed, False

    def run_text(
        prompt,
        target_length,
        steps,
        cfg,
        seed,
        randomize,
        stop,
        progress=gr.Progress(),
    ):
        selected_seed = _resolved_seed(seed, randomize)
        progress(0.05, desc="Generating continuous text embeddings")
        text, metadata = _safe_call(
            backend.continue_text,
            prompt,
            target_length,
            steps,
            cfg,
            selected_seed,
            stop,
        )
        progress(1.0, desc="Complete")
        return text, metadata, selected_seed, False

    with gr.Blocks(title="MF-1 · Multimodal Flow") as demo:
        gr.HTML(
            '<header class="mf-hero">'
            '<p class="mf-kicker">HUST Vision Lab · Research Demo</p>'
            '<h1 class="mf-title">Multimodal <span>Flow</span></h1>'
            '<p class="mf-subtitle">One continuous generative process for language '
            "and vision. Explore MF-1 understanding, generation, and text continuation "
            "through the official inference pipeline.</p>"
            f'<span class="mf-badge">{badge}</span>'
            "</header>"
        )

        with gr.Tabs():
            with gr.Tab("Text → Image"):
                gr.Markdown(
                    "Generate an image from a natural-language description.",
                    elem_classes="mf-lead",
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="mf-panel"):
                        image_prompt = gr.Textbox(
                            label="Image description",
                            lines=4,
                            placeholder="A quiet observatory above the clouds...",
                        )
                        gr.Examples(T2I_EXAMPLES, inputs=image_prompt, label="Examples")
                        with gr.Accordion("Generation controls", open=True):
                            image_steps = gr.Slider(
                                4,
                                128,
                                value=defaults["steps"],
                                step=4,
                                label="Flow steps",
                            )
                            image_cfg = gr.Slider(
                                0,
                                12,
                                value=defaults["cfg"],
                                step=0.25,
                                label="CFG scale",
                            )
                            image_method = gr.Radio(
                                ["sde", "ode"],
                                value=defaults["method"],
                                label="Sampler",
                            )
                            with gr.Row():
                                image_seed = gr.Number(
                                    value=42,
                                    precision=0,
                                    minimum=0,
                                    maximum=MAX_SEED,
                                    label="Seed",
                                )
                                image_random = gr.Checkbox(
                                    value=True,
                                    label="Randomize seed",
                                )
                        with gr.Row():
                            image_button = gr.Button("Generate image", variant="primary")
                            surprise_button = gr.Button("Surprise me")
                    with gr.Column(scale=6, elem_classes="mf-panel"):
                        image_output = gr.Image(
                            label="MF-1 output",
                            type="pil",
                            interactive=False,
                            height=500,
                        )
                        image_metadata = gr.JSON(label="Reproducibility metadata")

                image_outputs = [
                    image_output,
                    image_metadata,
                    image_seed,
                    image_random,
                ]
                image_button.click(
                    run_image,
                    [
                        image_prompt,
                        image_steps,
                        image_cfg,
                        image_method,
                        image_seed,
                        image_random,
                    ],
                    image_outputs,
                    api_name="generate_image",
                )
                image_prompt.submit(
                    run_image,
                    [
                        image_prompt,
                        image_steps,
                        image_cfg,
                        image_method,
                        image_seed,
                        image_random,
                    ],
                    image_outputs,
                    api_name=False,
                )
                surprise_button.click(
                    run_unconditional,
                    [image_steps, image_method, image_seed, image_random],
                    image_outputs,
                    api_name="generate_unconditional",
                )

            with gr.Tab("Image → Text"):
                gr.Markdown(
                    "Upload an image for captioning or visual question answering.",
                    elem_classes="mf-lead",
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="mf-panel"):
                        input_image = gr.Image(
                            label="Input image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=350,
                        )
                        question = gr.Textbox(
                            label="Question or instruction",
                            value="Describe this image in detail.",
                            lines=2,
                        )
                        with gr.Accordion("Generation controls", open=False):
                            caption_length = gr.Slider(
                                8, 256, value=128, step=8, label="Maximum answer length"
                            )
                            caption_steps = gr.Slider(4, 64, value=16, step=4, label="Flow steps")
                            caption_cfg = gr.Slider(
                                0, 12, value=defaults["cfg"], step=0.25, label="CFG scale"
                            )
                            with gr.Row():
                                caption_seed = gr.Number(
                                    value=42,
                                    precision=0,
                                    minimum=0,
                                    maximum=MAX_SEED,
                                    label="Seed",
                                )
                                caption_random = gr.Checkbox(
                                    value=False,
                                    label="Randomize seed",
                                )
                        caption_button = gr.Button("Analyze image", variant="primary")
                    with gr.Column(scale=6, elem_classes="mf-panel"):
                        caption_output = gr.Textbox(
                            label="MF-1 response",
                            lines=10,
                            interactive=False,
                        )
                        caption_metadata = gr.JSON(label="Reproducibility metadata")

                caption_button.click(
                    run_caption,
                    [
                        input_image,
                        question,
                        caption_length,
                        caption_steps,
                        caption_cfg,
                        caption_seed,
                        caption_random,
                    ],
                    [caption_output, caption_metadata, caption_seed, caption_random],
                    api_name="analyze_image",
                )

            with gr.Tab("Text continuation"):
                gr.Markdown(
                    "Continue a text prefix through MF-1's continuous text representation.",
                    elem_classes="mf-lead",
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="mf-panel"):
                        text_prompt = gr.Textbox(
                            label="Text prefix",
                            lines=5,
                            placeholder="A short language model can",
                        )
                        gr.Examples(TEXT_EXAMPLES, inputs=text_prompt, label="Examples")
                        with gr.Accordion("Generation controls", open=False):
                            text_length = gr.Slider(
                                8, 512, value=128, step=8, label="Maximum continuation length"
                            )
                            text_steps = gr.Slider(4, 64, value=16, step=4, label="Flow steps")
                            text_cfg = gr.Slider(
                                0, 12, value=defaults["cfg"], step=0.25, label="CFG scale"
                            )
                            stop_sequences = gr.Textbox(
                                label="Stop sequences (one per line)",
                                lines=2,
                                placeholder="Optional",
                            )
                            with gr.Row():
                                text_seed = gr.Number(
                                    value=42,
                                    precision=0,
                                    minimum=0,
                                    maximum=MAX_SEED,
                                    label="Seed",
                                )
                                text_random = gr.Checkbox(
                                    value=False,
                                    label="Randomize seed",
                                )
                        text_button = gr.Button("Continue text", variant="primary")
                    with gr.Column(scale=6, elem_classes="mf-panel"):
                        text_output = gr.Textbox(
                            label="MF-1 continuation",
                            lines=12,
                            interactive=False,
                        )
                        text_metadata = gr.JSON(label="Reproducibility metadata")

                text_outputs = [text_output, text_metadata, text_seed, text_random]
                text_button.click(
                    run_text,
                    [
                        text_prompt,
                        text_length,
                        text_steps,
                        text_cfg,
                        text_seed,
                        text_random,
                        stop_sequences,
                    ],
                    text_outputs,
                    api_name="continue_text",
                )
                text_prompt.submit(
                    run_text,
                    [
                        text_prompt,
                        text_length,
                        text_steps,
                        text_cfg,
                        text_seed,
                        text_random,
                        stop_sequences,
                    ],
                    text_outputs,
                    api_name=False,
                )

            with gr.Tab("System information"):
                gr.Markdown(
                    "Runtime details captured when the application loaded. Generated "
                    "results include their own timing and peak-memory metadata."
                )
                gr.JSON(value=status, label="MF-1 runtime")
                gr.Markdown(
                    "The production backend accepts one request at a time so GPU memory "
                    "and model state remain predictable."
                )

        gr.HTML(
            '<footer class="mf-footer">Built on the official '
            '<a href="https://github.com/hustvl/Multimodal-Flow" target="_blank">'
            "Multimodal Flow</a> inference pipeline. Mock outputs are always marked and "
            "must never be presented as model results.</footer>"
        )

    return demo


def launch_style() -> dict[str, Any]:
    return {
        "theme": theme(),
        "css": css(),
        "allowed_paths": [str((REPO_ROOT / "docs" / "assets").resolve())],
    }
