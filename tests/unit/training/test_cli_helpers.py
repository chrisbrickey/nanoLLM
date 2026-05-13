"""Unit tests for src/training/cli.py — argparse helpers and config builders
shared by scripts/train.py and scripts/resume.py."""

import argparse
from pathlib import Path

import pytest

from src.config import TrainingConfig
from src.paths import DEFAULT_DATA_FILE
from src.training.cli import (
    add_shared_training_args,
    build_training_config,
    resolve_data_file,
    resolve_destination_checkpoint,
)


def _parser_with_shared_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_shared_training_args(parser)
    return parser


class TestAddSharedArgs:
    def test_empty_argv_yields_all_none(self) -> None:
        """Every shared override must default to None so callers can detect
        'user did not pass this' and fall back to TrainingConfig defaults."""
        args = _parser_with_shared_args().parse_args([])
        assert args.batch_size is None
        assert args.epochs is None
        assert args.max_stories is None
        assert args.seed is None
        assert args.shuffle is None
        assert args.data_file is None
        assert args.checkpoint_destination is None

    def test_all_flags_parsed(self) -> None:
        argv = [
            "--batch-size", "8",
            "--epochs", "7",
            "--max-stories", "5",
            "--seed", "123",
            "--shuffle",
            "--data-file", "data/sample.txt",
            "--checkpoint-destination", "checkpoints/run_x",
        ]
        args = _parser_with_shared_args().parse_args(argv)
        assert args.batch_size == 8
        assert args.epochs == 7
        assert args.max_stories == 5
        assert args.seed == 123
        assert args.shuffle is True
        assert args.data_file == "data/sample.txt"
        assert args.checkpoint_destination == "checkpoints/run_x"

    def test_no_shuffle_flag_sets_false(self) -> None:
        args = _parser_with_shared_args().parse_args(["--no-shuffle"])
        assert args.shuffle is False


class TestBuildTrainingConfig:
    def test_empty_args_returns_defaults(self) -> None:
        args = _parser_with_shared_args().parse_args([])
        config = build_training_config(args)
        assert config == TrainingConfig()

    @pytest.mark.parametrize(
        "flag,value,field,expected",
        [
            ("--batch-size", "8", "batch_size", 8),
            ("--epochs", "7", "epochs", 7),
            ("--max-stories", "5", "max_stories", 5),
            ("--seed", "123", "seed", 123),
        ],
    )
    def test_single_flag_overrides_one_field(
        self, flag: str, value: str, field: str, expected: int
    ) -> None:
        args = _parser_with_shared_args().parse_args([flag, value])
        config = build_training_config(args)
        assert getattr(config, field) == expected

    def test_shuffle_flag_toggles_true(self) -> None:
        args = _parser_with_shared_args().parse_args(["--shuffle"])
        config = build_training_config(args)
        assert config.shuffle is True

    def test_no_shuffle_flag_toggles_false(self) -> None:
        args = _parser_with_shared_args().parse_args(["--no-shuffle"])
        config = build_training_config(args)
        assert config.shuffle is False

    def test_omitting_shuffle_uses_training_config_default(self) -> None:
        args = _parser_with_shared_args().parse_args([])
        config = build_training_config(args)
        assert config.shuffle == TrainingConfig().shuffle


class TestResolveDataFile:
    def test_returns_default_when_arg_is_none(self) -> None:
        args = _parser_with_shared_args().parse_args([])
        assert resolve_data_file(args) == DEFAULT_DATA_FILE

    def test_returns_supplied_path(self) -> None:
        args = _parser_with_shared_args().parse_args(["--data-file", "data/other.txt"])
        assert resolve_data_file(args) == Path("data/other.txt")


class TestResolveDestinationCheckpoint:
    def test_returns_timestamped_default_when_arg_is_none(self) -> None:
        args = _parser_with_shared_args().parse_args([])
        path = resolve_destination_checkpoint(args)
        # Default is the project-wide default_checkpoint_path("NanoLLM"); just
        # assert structure — the timestamp itself is verified elsewhere.
        assert "NanoLLM_" in path.name

    def test_returns_supplied_path(self) -> None:
        args = _parser_with_shared_args().parse_args(
            ["--checkpoint-destination", "checkpoints/my_run"]
        )
        assert resolve_destination_checkpoint(args) == Path("checkpoints/my_run")
