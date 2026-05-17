"""Unit tests for src/training/schema.py — MetricsHistory and ResumeContext."""

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from src.checkpoint import CheckpointMetadata
from src.training.schema import MetricsHistory, ResumeContext


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAMPLE_SOURCE = Path("/fake/checkpoints/sample_bundle")
SAMPLE_PRIOR_EPOCHS = 7


# ---------------------------------------------------------------------------
# MetricsHistory
# ---------------------------------------------------------------------------

class TestMetricsHistoryDefaults:
    def test_train_loss_starts_empty(self) -> None:
        assert MetricsHistory().train_loss == []

    def test_final_train_loss_is_none_when_empty(self) -> None:
        assert MetricsHistory().final_train_loss is None


class TestMetricsHistoryRecord:
    def test_record_loss_appends_to_train_loss(self) -> None:
        h = MetricsHistory()
        h.record("loss", 0.5)
        assert h.train_loss == [0.5]

    def test_record_accumulates_values_in_order(self) -> None:
        h = MetricsHistory()
        h.record("loss", 0.9)
        h.record("loss", 0.6)
        h.record("loss", 0.3)
        assert h.train_loss == [0.9, 0.6, 0.3]

    def test_record_unknown_metric_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            MetricsHistory().record("accuracy", 0.9)


class TestMetricsHistoryFinalTrainLoss:
    def test_returns_last_value(self) -> None:
        h = MetricsHistory(train_loss=[0.9, 0.6, 0.3])
        assert h.final_train_loss == 0.3

    def test_returns_none_when_empty(self) -> None:
        assert MetricsHistory(train_loss=[]).final_train_loss is None

    def test_returns_only_value_for_single_entry(self) -> None:
        assert MetricsHistory(train_loss=[0.7]).final_train_loss == 0.7


class TestMetricsHistoryEquality:
    def test_two_empty_instances_are_equal(self) -> None:
        assert MetricsHistory() == MetricsHistory()

    def test_instances_with_same_values_are_equal(self) -> None:
        assert MetricsHistory(train_loss=[0.5, 0.3]) == MetricsHistory(train_loss=[0.5, 0.3])

    def test_instances_with_different_values_are_not_equal(self) -> None:
        assert MetricsHistory(train_loss=[0.5]) != MetricsHistory()


# ---------------------------------------------------------------------------
# ResumeContext
# ---------------------------------------------------------------------------

class TestResumeContextFromCheckpoint:
    def test_pulls_cumulative_epochs_from_metadata(self) -> None:
        sample_metadata = CheckpointMetadata(
            cumulative_epochs_completed=SAMPLE_PRIOR_EPOCHS,
        )
        with patch("src.training.schema.load_metadata", return_value=sample_metadata) as mock_load:
            ctx = ResumeContext.from_checkpoint(SAMPLE_SOURCE)

        mock_load.assert_called_once_with(SAMPLE_SOURCE)
        assert ctx.source == SAMPLE_SOURCE
        assert ctx.previous_epochs_completed == SAMPLE_PRIOR_EPOCHS

    def test_raises_when_metadata_missing(self) -> None:
        with patch("src.training.schema.load_metadata", return_value=None):
            with pytest.raises(ValueError, match="no readable metadata"):
                ResumeContext.from_checkpoint(SAMPLE_SOURCE)


class TestResumeContextImmutability:
    def test_dataclass_is_frozen(self) -> None:
        ctx = ResumeContext(source=SAMPLE_SOURCE, previous_epochs_completed=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.previous_epochs_completed = 2  # type: ignore[misc]
