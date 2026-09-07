"""Unit tests for src/training/schema.py"""

from datetime import datetime

import pytest

from src.training.schema import CheckpointMetadata, MetricsHistory


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAMPLE_CUMULATIVE_EPOCHS = 7
SAMPLE_FINAL_LOSS = 0.42
SAMPLE_MODEL_CONFIG = {"embed_dim": 12}
SAMPLE_TRAINING_CONFIG = {"epochs": 7}
SAMPLE_TOKENIZER_CONFIG = {"name": "gpt2"}


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
# CheckpointMetadata
# ---------------------------------------------------------------------------

class TestCheckpointMetadataDefaults:
    def test_minimum_construction_only_requires_cumulative_epochs(self) -> None:
        meta = CheckpointMetadata(cumulative_epochs_completed=SAMPLE_CUMULATIVE_EPOCHS)
        assert meta.cumulative_epochs_completed == SAMPLE_CUMULATIVE_EPOCHS
        assert meta.final_loss is None
        assert meta.model_config is None
        assert meta.training_config is None
        assert meta.tokenizer_config is None

    def test_created_at_defaults_to_an_iso_timestamp(self) -> None:
        meta = CheckpointMetadata(cumulative_epochs_completed=1)
        # Should parse as ISO 8601 without raising
        datetime.fromisoformat(meta.created_at)


class TestCheckpointMetadataFields:
    def test_records_all_provided_fields(self) -> None:
        meta = CheckpointMetadata(
            cumulative_epochs_completed=SAMPLE_CUMULATIVE_EPOCHS,
            final_loss=SAMPLE_FINAL_LOSS,
            model_config=SAMPLE_MODEL_CONFIG,
            training_config=SAMPLE_TRAINING_CONFIG,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        )
        assert meta.final_loss == SAMPLE_FINAL_LOSS
        assert meta.model_config == SAMPLE_MODEL_CONFIG
        assert meta.training_config == SAMPLE_TRAINING_CONFIG
        assert meta.tokenizer_config == SAMPLE_TOKENIZER_CONFIG
