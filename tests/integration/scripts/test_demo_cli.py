"""Integration tests for scripts/demo.py CLI

demo.launch is mocked to prevent server spinup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from scripts.demo import main
from src.config import InferenceConfig
from src.inference.demo import build_demo as real_build_demo
from tests.conftest import MakeRunCli, RunCli, assert_error_exit
from tests.integration.scripts.conftest import (
    CaptureCall,
    CheckpointResolutionErrorTests,
)

PROG = "nanollm-demo"
LOGGER_NAME = "scripts.demo"
BUILD_DEMO_TARGET = "scripts.demo.build_demo"
MAX_NEW_TOKENS_LABEL = "Max new tokens"
TEMPERATURE_LABEL = "Temperature"


@pytest.fixture
def run_cli(make_run_cli: MakeRunCli) -> RunCli:
    return make_run_cli(main, PROG)


@pytest.fixture
def mock_demo() -> MagicMock:
    """A stand-in gr.Interface whose launch never opens a server."""
    return MagicMock()


def _slider_by_label(demo: gr.Interface, label: str) -> gr.Slider:
    """Pull the real slider component with the given label off a built demo."""
    for component in demo.input_components:
        if isinstance(component, gr.Slider) and component.label == label:
            return component
    raise AssertionError(f"no slider found with label {label!r}")


class TestCliHappyPath:
    def test_explicit_checkpoint_source_launches_demo(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        mock_demo: MagicMock,
    ) -> None:
        with patch(BUILD_DEMO_TARGET, return_value=mock_demo) as mock_build:
            run_cli(["--checkpoint-source", str(checkpoint_bundle)])

        mock_build.assert_called_once()
        mock_demo.launch.assert_called_once_with()

    def test_falls_back_to_latest_checkpoint_when_source_omitted(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        mock_demo: MagicMock,
    ) -> None:
        with patch(BUILD_DEMO_TARGET, return_value=mock_demo), \
             patch("src.cli.get_latest_checkpoint", return_value=checkpoint_bundle):
            run_cli()

        mock_demo.launch.assert_called_once_with()


class TestCliEndToEnd:
    """Run the real checkpoint restore and demo construction; only launch is stubbed."""

    def test_builds_and_launches_real_demo_without_starting_a_server(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        with patch("gradio.Interface.launch") as mock_launch:
            run_cli(["--checkpoint-source", str(checkpoint_bundle)])

        mock_launch.assert_called_once_with()

    def test_real_slider_extremes_are_all_accepted_by_inference_config(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        capture_call: CaptureCall,
    ) -> None:
        """Whatever the real tokens/temperature sliders can emit at their extremes must be
        constructible InferenceConfig values; a slider bound drifting past a hard limit
        would otherwise only surface as a runtime error from a live demo submission."""
        with patch("gradio.Interface.launch"), \
             capture_call(BUILD_DEMO_TARGET, wrapping=real_build_demo) as call:
            run_cli(["--checkpoint-source", str(checkpoint_bundle)])

        demo = call.result
        assert isinstance(demo, gr.Interface)
        tokens_slider = _slider_by_label(demo, MAX_NEW_TOKENS_LABEL)
        temperature_slider = _slider_by_label(demo, TEMPERATURE_LABEL)

        for max_new_tokens in (tokens_slider.minimum, tokens_slider.maximum):
            for temperature in (temperature_slider.minimum, temperature_slider.maximum):
                InferenceConfig(max_new_tokens=int(max_new_tokens), temperature=temperature)


class TestCliErrors(CheckpointResolutionErrorTests):
    CAPTURED_LOGGER = LOGGER_NAME

    def test_build_demo_failure_exits_1(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch(BUILD_DEMO_TARGET, side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                run_cli(["--checkpoint-source", str(checkpoint_bundle)])
        assert_error_exit(exc_info, caplog)
