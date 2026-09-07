"""Unit tests for src/inference/cli.py"""

import argparse

import pytest

from src.config import InferenceConfig
from src.inference.cli import add_inference_args, build_inference_config

SAMPLE_PROMPT = "sample-text"
SAMPLE_CHECKPOINT_PATH = "checkpoints/sample_bundle"


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

