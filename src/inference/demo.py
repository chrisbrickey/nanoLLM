"""
nanoLLM/src/inference/demo.py

Builds a Gradio web demo for interactive text generation.
This script hooks into the existing inference pathway.
"""

from collections.abc import Callable

import gradio as gr

from src.config import (
    RECOMMENDED_NEW_TOKENS,
    RECOMMENDED_TEMPERATURE,
    InferenceConfig,
    TokenizerConfig,
)
from src.inference.completion import complete_prompt
from src.model.model import NanoLLM

DemoFn = Callable[[str, float, float], str]

INVALID_INPUT_PREFIX = "Invalid input: "

# Widget granularity only. Slider bounds come from the recommended bands in config.
SLIDER_NEW_TOKENS_STEP = 1
SLIDER_TEMPERATURE_STEP = 0.01


def make_demo_fn(*, model: NanoLLM, tokenizer_config: TokenizerConfig) -> DemoFn:
    """Build the closure that Gradio calls on every submit."""

    def _demo_fn(prompt: str, max_new_tokens: float, temperature: float) -> str:
        try:
            inference_config = InferenceConfig(
                max_new_tokens=int(max_new_tokens), temperature=temperature
            )
        except ValueError as e:
            # Gradio enforces slider bounds server-side, but this still guards
            # direct programmatic callers of make_demo_fn that bypass the sliders.
            return f"{INVALID_INPUT_PREFIX}{e}"

        return complete_prompt(
            model=model,
            tokenizer_config=tokenizer_config,
            inference_config=inference_config,
            prompt=prompt,
        )

    return _demo_fn


def build_demo(*, model: NanoLLM, tokenizer_config: TokenizerConfig) -> gr.Interface:
    """Assemble the Gradio Interface for the text generation demo."""
    return gr.Interface(
        fn=make_demo_fn(model=model, tokenizer_config=tokenizer_config),
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Slider(
                minimum=RECOMMENDED_NEW_TOKENS.minimum,
                maximum=RECOMMENDED_NEW_TOKENS.maximum,
                value=InferenceConfig.max_new_tokens,
                step=SLIDER_NEW_TOKENS_STEP,
                label="Max new tokens",
            ),
            gr.Slider(
                minimum=RECOMMENDED_TEMPERATURE.minimum,
                maximum=RECOMMENDED_TEMPERATURE.maximum,
                value=InferenceConfig.temperature,
                step=SLIDER_TEMPERATURE_STEP,
                label="Temperature",
            ),
        ],
        outputs=gr.Textbox(label="Generated text"),
    )
