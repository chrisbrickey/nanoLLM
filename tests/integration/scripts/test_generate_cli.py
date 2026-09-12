"""Integration tests for scripts/generate.py CLI"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.generate import main
from src.config import InferenceConfig
from tests.conftest import (
    ABOVE_RANGE_TEMPERATURE,
    BELOW_RANGE_TEMPERATURE,
    SAMPLE_COMPLETION,
    SAMPLE_PROMPT,
    MakeRunCli,
    RunCli,
    assert_error_exit,
)
from tests.integration.scripts.conftest import (
    CaptureCall,
    CheckpointResolutionErrorTests,
)

PROG = "nanollm-generate"
LOGGER_NAME = "scripts.generate"
INFERENCE_CLI_LOGGER_NAME = "src.inference.cli"
COMPLETE_PROMPT_TARGET = "scripts.generate.complete_prompt"

SAMPLE_MAX_NEW_TOKENS = 7
SAMPLE_TEMPERATURE = 0.5
SAMPLE_SEED = 42

# A small token budget and fixed seed keep an unmocked run fast and repeatable
E2E_ARGS = ["--max-new-tokens", "3", "--seed", "0"]


@pytest.fixture
def run_cli(make_run_cli: MakeRunCli) -> RunCli:
    """Run the generate CLI and return everything it printed.

    Every invocation needs a prompt, so a sample one is supplied unless the caller
    passes its own.
    """
    runner = make_run_cli(main, PROG)

    def _run(args: list[str] | None = None) -> str:
        supplied = list(args or [])
        if "--prompt" not in supplied:
            supplied = ["--prompt", SAMPLE_PROMPT, *supplied]
        return runner(supplied)

    return _run


class TestCliHappyPath:
    def test_explicit_checkpoint_source_prints_completion(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        with patch(COMPLETE_PROMPT_TARGET, return_value=SAMPLE_COMPLETION):
            out = run_cli(["--checkpoint-source", str(checkpoint_bundle)])

        assert SAMPLE_COMPLETION in out

    def test_falls_back_to_latest_checkpoint_when_source_omitted(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        with patch(COMPLETE_PROMPT_TARGET, return_value=SAMPLE_COMPLETION), \
             patch("src.cli.get_latest_checkpoint", return_value=checkpoint_bundle):
            out = run_cli()

        assert SAMPLE_COMPLETION in out


class TestCliEndToEnd:
    """Run the full pipeline (checkpoint restore, encode, generate, decode) with no mocks."""

    def test_generates_and_prints_completion_containing_prompt(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        out = run_cli(["--checkpoint-source", str(checkpoint_bundle), *E2E_ARGS])
        assert "GENERATED TEXT" in out
        assert SAMPLE_PROMPT in out

    def test_same_seed_produces_identical_output(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        args = ["--checkpoint-source", str(checkpoint_bundle), *E2E_ARGS]
        first = run_cli(args)
        second = run_cli(args)
        assert first == second


class TestCliErrors(CheckpointResolutionErrorTests):
    CAPTURED_LOGGER = LOGGER_NAME

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_empty_prompt_exits_2_without_loading_a_checkpoint(
        self,
        run_cli: RunCli,
        prompt: str,
    ) -> None:
        """An unusable prompt is a usage error, so argparse rejects it before the model loads."""
        with patch("scripts.generate.restore_from_checkpoint") as mock_restore:
            with pytest.raises(SystemExit) as exc_info:
                run_cli(["--prompt", prompt])
        assert exc_info.value.code == 2
        mock_restore.assert_not_called()

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--max-new-tokens", "0"),
            ("--temperature", "0"),
        ],
    )
    def test_invalid_flag_values_exit_1(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
        flag: str,
        value: str,
    ) -> None:
        """Config validation fails before any checkpoint is read, so none is needed here."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli([flag, value])
        assert_error_exit(exc_info, caplog)


class TestCliFlags:
    def test_flags_and_prompt_propagate(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        capture_call: CaptureCall,
    ) -> None:
        with capture_call(COMPLETE_PROMPT_TARGET, returning=SAMPLE_COMPLETION) as call:
            out = run_cli([
                "--checkpoint-source", str(checkpoint_bundle),
                "--max-new-tokens", str(SAMPLE_MAX_NEW_TOKENS),
                "--temperature", str(SAMPLE_TEMPERATURE),
                "--seed", str(SAMPLE_SEED),
            ])

        assert call.kwargs["prompt"] == SAMPLE_PROMPT
        config = call.kwargs["inference_config"]
        assert isinstance(config, InferenceConfig)
        assert config.max_new_tokens == SAMPLE_MAX_NEW_TOKENS
        assert config.temperature == SAMPLE_TEMPERATURE
        assert config.seed == SAMPLE_SEED

        assert SAMPLE_COMPLETION in out


class TestCliTemperatureRangeWarning:
    """The RECOMMENDED_TEMPERATURE band is advisory. The behavior under test is that
    the generate.py path surfaces the warning and still generates an unclamped output."""

    CAPTURED_LOGGER = INFERENCE_CLI_LOGGER_NAME
    CAPTURED_LEVEL = logging.WARNING

    @pytest.mark.parametrize(
        "temperature",
        [ABOVE_RANGE_TEMPERATURE, BELOW_RANGE_TEMPERATURE],
    )
    def test_out_of_band_temperature_warns_but_still_generates_unclamped(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
        capture_call: CaptureCall,
        temperature: float,
    ) -> None:
        with capture_call(COMPLETE_PROMPT_TARGET, returning=SAMPLE_COMPLETION) as call:
            out = run_cli([
                "--checkpoint-source", str(checkpoint_bundle),
                "--temperature", str(temperature),
            ])

        assert any(
            r.levelno == logging.WARNING and r.name == INFERENCE_CLI_LOGGER_NAME
            for r in caplog.records
        )
        config = call.kwargs["inference_config"]
        assert isinstance(config, InferenceConfig)
        assert config.temperature == temperature
        assert SAMPLE_COMPLETION in out
