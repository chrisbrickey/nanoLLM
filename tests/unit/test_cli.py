"""Unit tests for src/cli.py"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import apply_cli_overrides, resolve_source_checkpoint

DEFAULT_COUNT = 3
DEFAULT_LABEL = "sample-label"
SAMPLE_CHECKPOINT_PATH = "checkpoints/sample_bundle"


@dataclass(frozen=True)
class SampleConfig:
    count: int = DEFAULT_COUNT
    label: str = DEFAULT_LABEL
    ratio: float = 0.5


def _args(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


class TestApplyCliOverrides:
    def test_all_none_returns_defaults(self) -> None:
        result = apply_cli_overrides(SampleConfig(), _args(count=None, label=None))
        assert result == SampleConfig()

    def test_non_none_values_override_defaults(self) -> None:
        result = apply_cli_overrides(SampleConfig(), _args(count=42, label=None))
        assert result.count == 42
        assert result.label == DEFAULT_LABEL

    def test_args_that_do_not_name_a_config_field_are_ignored(self) -> None:
        result = apply_cli_overrides(SampleConfig(), _args(count=None, data_file="data/sample.txt"))
        assert result == SampleConfig()

    def test_fields_without_a_matching_arg_keep_their_default(self) -> None:
        """The config declares 'ratio' but no flag supplies it."""
        result = apply_cli_overrides(SampleConfig(), _args(count=7))
        assert result.ratio == SampleConfig().ratio

    def test_source_config_is_not_mutated(self) -> None:
        original = SampleConfig()
        apply_cli_overrides(original, _args(count=99))
        assert original.count == DEFAULT_COUNT

    def test_invalid_override_propagates_config_validation_error(self) -> None:
        @dataclass(frozen=True)
        class ValidatedConfig:
            count: int = DEFAULT_COUNT

            def __post_init__(self) -> None:
                if self.count <= 0:
                    raise ValueError(f"count must be > 0, got {self.count}")

        with pytest.raises(ValueError, match="count"):
            apply_cli_overrides(ValidatedConfig(), _args(count=0))


class TestResolveSourceCheckpoint:
    def test_returns_supplied_path(self) -> None:
        args = _args(checkpoint_source=SAMPLE_CHECKPOINT_PATH)
        assert resolve_source_checkpoint(args) == Path(SAMPLE_CHECKPOINT_PATH)

    def test_falls_back_to_latest_checkpoint_when_arg_is_none(self) -> None:
        args = _args(checkpoint_source=None)
        latest = Path(SAMPLE_CHECKPOINT_PATH)
        with patch("src.cli.get_latest_checkpoint", return_value=latest) as mock_latest:
            result = resolve_source_checkpoint(args)
        assert result == latest
        mock_latest.assert_called_once()

    def test_raises_file_not_found_when_no_checkpoint_exists(self) -> None:
        args = _args(checkpoint_source=None)
        with patch("src.cli.get_latest_checkpoint", return_value=None):
            with pytest.raises(FileNotFoundError):
                resolve_source_checkpoint(args)
