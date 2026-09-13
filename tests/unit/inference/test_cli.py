"""Unit tests for src/inference/cli.py"""

import argparse
import logging

import pytest

from src.config import InferenceConfig
from src.inference.cli import add_inference_args, build_inference_config
from tests.conftest import (
    ABOVE_RANGE_NEW_TOKENS,
    ABOVE_RANGE_TEMPERATURE,
    BELOW_RANGE_TEMPERATURE,
    IN_RANGE_NEW_TOKENS,
    IN_RANGE_TEMPERATURE,
    SAMPLE_PROMPT,
)

SAMPLE_CHECKPOINT_PATH = "checkpoints/sample_bundle"
CLI_LOGGER = "src.inference.cli"


def _parser_with_inference_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    return parser


class TestAddInferenceArgs:
    def test_empty_argv_with_prompt_yields_none_for_optional_flags(self) -> None:
        args = _parser_with_inference_args().parse_args(["--prompt", SAMPLE_PROMPT])
        assert args.prompt == SAMPLE_PROMPT
        assert args.checkpoint_source is None
        assert args.max_new_tokens is None
        assert args.temperature is None
        assert args.seed is None

    def test_omitting_prompt_exits_2(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parser_with_inference_args().parse_args([])
        assert exc_info.value.code == 2

    def test_all_flags_parsed(self) -> None:
        argv = [
            "--prompt", SAMPLE_PROMPT,
            "--checkpoint-source", SAMPLE_CHECKPOINT_PATH,
            "--max-new-tokens", "7",
            "--temperature", "0.5",
            "--seed", "42",
        ]
        args = _parser_with_inference_args().parse_args(argv)
        assert args.prompt == SAMPLE_PROMPT
        assert args.checkpoint_source == SAMPLE_CHECKPOINT_PATH
        assert args.max_new_tokens == 7
        assert args.temperature == 0.5
        assert args.seed == 42


class TestBuildInferenceConfig:
    def test_empty_args_returns_defaults(self) -> None:
        args = _parser_with_inference_args().parse_args(["--prompt", SAMPLE_PROMPT])
        config = build_inference_config(args)
        assert config == InferenceConfig()

    @pytest.mark.parametrize(
        "flag,value,field,expected",
        [
            ("--max-new-tokens", "7", "max_new_tokens", 7),
            ("--temperature", "0.5", "temperature", 0.5),
            ("--seed", "42", "seed", 42),
        ],
    )
    def test_single_flag_overrides_one_field(
        self, flag: str, value: str, field: str, expected: object
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, flag, value]
        )
        config = build_inference_config(args)
        assert getattr(config, field) == expected

    def test_invalid_max_new_tokens_raises_value_error(self) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--max-new-tokens", "0"]
        )
        with pytest.raises(ValueError):
            build_inference_config(args)

    def test_invalid_temperature_raises_value_error(self) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--temperature", "0"]
        )
        with pytest.raises(ValueError):
            build_inference_config(args)


class TestBuildInferenceConfigTemperatureWarnings:
    """RECOMMENDED_TEMPERATURE is only a recommendation: out-of-range values warn but are never blocked or clamped."""

    def test_warns_when_temperature_below_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--temperature", str(BELOW_RANGE_TEMPERATURE)]
        )

        with caplog.at_level(logging.WARNING, logger=CLI_LOGGER):
            config = build_inference_config(args)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert config.temperature == BELOW_RANGE_TEMPERATURE

    def test_warns_when_temperature_above_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--temperature", str(ABOVE_RANGE_TEMPERATURE)]
        )

        with caplog.at_level(logging.WARNING, logger=CLI_LOGGER):
            config = build_inference_config(args)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert config.temperature == ABOVE_RANGE_TEMPERATURE

    def test_no_warning_when_temperature_inside_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--temperature", str(IN_RANGE_TEMPERATURE)]
        )

        with caplog.at_level(logging.WARNING, logger=CLI_LOGGER):
            config = build_inference_config(args)

        assert not any(r.levelno == logging.WARNING for r in caplog.records)
        assert config.temperature == IN_RANGE_TEMPERATURE

    def test_does_not_raise_for_out_of_range_temperature(self) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--temperature", str(ABOVE_RANGE_TEMPERATURE)]
        )

        config = build_inference_config(args)

        assert isinstance(config, InferenceConfig)



class TestBuildInferenceConfigMaxNewTokensWarnings:
    """RECOMMENDED_NEW_TOKENS is only a recommendation: out-of-range values warn but are never blocked or clamped."""

    def test_warns_when_max_new_tokens_above_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--max-new-tokens", str(ABOVE_RANGE_NEW_TOKENS)]
        )

        with caplog.at_level(logging.WARNING, logger=CLI_LOGGER):
            config = build_inference_config(args)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert config.max_new_tokens == ABOVE_RANGE_NEW_TOKENS

    def test_no_warning_when_max_new_tokens_inside_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        args = _parser_with_inference_args().parse_args(
            ["--prompt", SAMPLE_PROMPT, "--max-new-tokens", str(IN_RANGE_NEW_TOKENS)]
        )

        with caplog.at_level(logging.WARNING, logger=CLI_LOGGER):
            config = build_inference_config(args)

        assert not any(r.levelno == logging.WARNING for r in caplog.records)
        assert config.max_new_tokens == IN_RANGE_NEW_TOKENS
