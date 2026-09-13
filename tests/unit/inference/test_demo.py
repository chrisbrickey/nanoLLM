"""Unit tests for src/inference/demo.py"""

from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from src.config import (
    RECOMMENDED_NEW_TOKENS,
    RECOMMENDED_TEMPERATURE,
    InferenceConfig,
    TokenizerConfig,
)
from src.inference.demo import (
    INVALID_INPUT_PREFIX,
    SLIDER_NEW_TOKENS_STEP,
    SLIDER_TEMPERATURE_STEP,
    DemoFn,
    build_demo,
    make_demo_fn,
)
from tests.conftest import SAMPLE_COMPLETION, SAMPLE_PROMPT

# Gradio sliders emit floats, so the closure receives one where InferenceConfig needs an int
SAMPLE_MAX_NEW_TOKENS_FLOAT = 7.0
SAMPLE_MAX_NEW_TOKENS_INT = 7
SAMPLE_TEMPERATURE = 0.8


@pytest.fixture
def demo_fn(mock_model: MagicMock, tokenizer_config: TokenizerConfig) -> DemoFn:
    return make_demo_fn(model=mock_model, tokenizer_config=tokenizer_config)


class TestMakeDemoFn:
    def test_forwards_ui_inputs_to_complete_prompt_and_returns_its_result(
        self,
        demo_fn: DemoFn,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
    ) -> None:
        """The closure's whole job: turn slider values into one complete_prompt call."""
        with patch(
            "src.inference.demo.complete_prompt", return_value=SAMPLE_COMPLETION
        ) as mock_complete:
            result = demo_fn(
                SAMPLE_PROMPT, SAMPLE_MAX_NEW_TOKENS_FLOAT, SAMPLE_TEMPERATURE
            )

        assert result == SAMPLE_COMPLETION
        mock_complete.assert_called_once()
        _, kwargs = mock_complete.call_args
        assert kwargs["model"] is mock_model
        assert kwargs["tokenizer_config"] is tokenizer_config
        assert kwargs["prompt"] == SAMPLE_PROMPT

        config = kwargs["inference_config"]
        assert isinstance(config, InferenceConfig)
        assert config.max_new_tokens == SAMPLE_MAX_NEW_TOKENS_INT
        assert isinstance(config.max_new_tokens, int)
        assert config.temperature == SAMPLE_TEMPERATURE
        assert config.seed is None

    @pytest.mark.parametrize("temperature", [0.0, -0.5])
    def test_returns_friendly_message_for_out_of_range_temperature(
        self,
        demo_fn: DemoFn,
        temperature: float,
    ) -> None:
        """Slider bounds are client-side only, so a direct API call can still send a bad value."""
        with patch("src.inference.demo.complete_prompt") as mock_complete:
            result = demo_fn(SAMPLE_PROMPT, SAMPLE_MAX_NEW_TOKENS_FLOAT, temperature)

        assert result.startswith(INVALID_INPUT_PREFIX)
        assert "temperature" in result
        mock_complete.assert_not_called()


class TestBuildDemo:
    @pytest.fixture
    def demo(
        self,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
    ) -> gr.Interface:
        return build_demo(model=mock_model, tokenizer_config=tokenizer_config)

    def test_returns_gradio_interface(self, demo: gr.Interface) -> None:
        assert isinstance(demo, gr.Interface)

    def test_inputs_are_textbox_then_two_sliders_in_order(
        self, demo: gr.Interface
    ) -> None:
        prompt_input, tokens_slider, temperature_slider = demo.input_components
        assert isinstance(prompt_input, gr.Textbox)
        assert isinstance(tokens_slider, gr.Slider)
        assert isinstance(temperature_slider, gr.Slider)

    def test_max_tokens_slider_has_expected_bounds(self, demo: gr.Interface) -> None:
        _, tokens_slider, _ = demo.input_components
        assert tokens_slider.minimum == RECOMMENDED_NEW_TOKENS.minimum
        assert tokens_slider.maximum == RECOMMENDED_NEW_TOKENS.maximum
        assert tokens_slider.value == InferenceConfig.max_new_tokens
        assert tokens_slider.step == SLIDER_NEW_TOKENS_STEP

    def test_temperature_slider_has_expected_bounds(self, demo: gr.Interface) -> None:
        _, _, temperature_slider = demo.input_components
        assert temperature_slider.minimum == RECOMMENDED_TEMPERATURE.minimum
        assert temperature_slider.maximum == RECOMMENDED_TEMPERATURE.maximum
        assert temperature_slider.value == InferenceConfig.temperature
        assert temperature_slider.step == SLIDER_TEMPERATURE_STEP

    def test_has_single_text_output(self, demo: gr.Interface) -> None:
        assert len(demo.output_components) == 1
        assert isinstance(demo.output_components[0], gr.Textbox)

    def test_fn_delegates_to_complete_prompt(
        self,
        demo: gr.Interface,
    ) -> None:
        """The Interface is wired to the closure, not to some other callable."""
        with patch(
            "src.inference.demo.complete_prompt", return_value=SAMPLE_COMPLETION
        ) as mock_complete:
            result = demo.fn(
                SAMPLE_PROMPT, SAMPLE_MAX_NEW_TOKENS_FLOAT, SAMPLE_TEMPERATURE
            )

        assert result == SAMPLE_COMPLETION
        mock_complete.assert_called_once()
