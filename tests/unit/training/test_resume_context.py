"""Unit tests for src/training/resume_context.py — ResumeContext factory + immutability."""

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from src.checkpoint import CheckpointMetadata
from src.training.resume_context import ResumeContext


SAMPLE_SOURCE = Path("/fake/checkpoints/sample_bundle")
SAMPLE_PRIOR_EPOCHS = 7


class TestResumeContextFromCheckpoint:
    def test_pulls_cumulative_epochs_from_metadata(self) -> None:
        sample_metadata = CheckpointMetadata(
            cumulative_epochs_completed=SAMPLE_PRIOR_EPOCHS,
        )
        with patch("src.training.resume_context.load_metadata", return_value=sample_metadata) as mock_load:
            ctx = ResumeContext.from_checkpoint(SAMPLE_SOURCE)

        mock_load.assert_called_once_with(SAMPLE_SOURCE)
        assert ctx.source == SAMPLE_SOURCE
        assert ctx.previous_epochs_completed == SAMPLE_PRIOR_EPOCHS

    def test_raises_when_metadata_missing(self) -> None:
        with patch("src.training.resume_context.load_metadata", return_value=None):
            with pytest.raises(ValueError, match="no readable metadata"):
                ResumeContext.from_checkpoint(SAMPLE_SOURCE)


class TestResumeContextImmutability:
    def test_dataclass_is_frozen(self) -> None:
        ctx = ResumeContext(source=SAMPLE_SOURCE, previous_epochs_completed=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.previous_epochs_completed = 2  # type: ignore[misc]
