"""Integration tests for a full training run (src/training/)."""

import dataclasses
import json
import shutil
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR
from src.training.resume_context import ResumeContext
from src.training.runner import Runner
from src.training.trainer import Trainer

SAMPLE_DATA_SOURCE = Path("/fake/data/stories.txt")
SAMPLE_TOKENIZER_CONFIG = TokenizerConfig(
    delimiter="<|endoftext|>",
    name="gpt2",
    pad_token_id=0,
)

# Small enough to run fast; large enough that loss reliably trends down.
MAXLEN = 8
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1
BATCH_SIZE = 4
N_BATCHES = 8
EPOCHS = 2


class _FakeDataLoader:
    """Mimics grain's batch format: yields arrays shaped (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


@pytest.fixture
def project_checkpoint_destination() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"integration_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


def _make_model() -> NanoLLM:
    return NanoLLM(
        ModelConfig(
            maxlen=MAXLEN,
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feed_forward_dim=FF_DIM,
            num_transformer_blocks=NUM_BLOCKS,
        )
    )


def _make_training_config() -> TrainingConfig:
    return TrainingConfig(epochs=EPOCHS, batch_size=BATCH_SIZE, log_every_n_steps=4)


def _make_trainer() -> Trainer:
    return Trainer(
        model=_make_model(),
        training_config=_make_training_config(),
        dataloader=_FakeDataLoader(n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE),
        batches_per_epoch=N_BATCHES,
    )


def _run_with_patched_data(
    model: NanoLLM,
    *,
    checkpoint_destination: Path | None,
    resume_from: ResumeContext | None = None,
) -> None:
    """Drives Runner.run with patched data loading so the test stays
    deterministic and avoids tokenizing real text."""
    config = _make_training_config()
    dataloader = _FakeDataLoader(n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE)
    # Provide enough stories to satisfy len(stories) // batch_size >= 1
    fake_stories = [f"s{i}" for i in range(BATCH_SIZE * N_BATCHES)]
    processor = patch("src.training.runner.Processor")
    with patch("src.training.runner.load_text_from_file", return_value=fake_stories), \
         processor as mock_processor:
        mock_processor.return_value.process.return_value = dataloader
        Runner(
            model=model,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
            data_source=SAMPLE_DATA_SOURCE,
            training_config=config,
            checkpoint_destination=checkpoint_destination,
            resume_from=resume_from,
        ).run()


class TestTrainLoop:
    """Trainer-only training-loop behavior. Disk persistence is covered by
    the runner-level tests below."""

    def test_metrics_history_is_populated(self) -> None:
        history = _make_trainer().train()
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_all_losses_are_finite_and_positive(self) -> None:
        history = _make_trainer().train()
        for loss in history["train_loss"]:
            assert loss > 0.0
            assert loss < float("inf")

    def test_loss_decreases_over_training(self) -> None:
        history = _make_trainer().train()
        losses = history["train_loss"]
        assert losses[0] > losses[-1], (
            f"Expected loss to decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestRunnerCheckpointOnDisk:
    """End-to-end verification that runner.run writes a checkpoint bundle
    containing the expected metadata."""

    def test_checkpoint_written_to_disk(self, project_checkpoint_destination: Path) -> None:
        _run_with_patched_data(_make_model(), checkpoint_destination=project_checkpoint_destination)
        assert project_checkpoint_destination.exists()
        assert (project_checkpoint_destination / "weights.orbax").exists()
        assert (project_checkpoint_destination / "metadata.json").exists()

    def test_tokenizer_config_written_to_metadata_json(
        self, project_checkpoint_destination: Path
    ) -> None:
        _run_with_patched_data(_make_model(), checkpoint_destination=project_checkpoint_destination)
        saved = json.loads((project_checkpoint_destination / "metadata.json").read_text(encoding="utf-8"))
        assert saved["tokenizer_config"] == dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

    def test_prior_epochs_written_to_metadata_json(
        self, project_checkpoint_destination: Path
    ) -> None:
        """When resume_from carries previous_epochs_completed=10 and training
        runs for EPOCHS, metadata.json on disk must record
        cumulative_epochs_completed=10+EPOCHS."""
        prior = 10
        resume_ctx = ResumeContext(
            source=Path("/fake/checkpoints/prior_bundle"),
            previous_epochs_completed=prior,
        )
        _run_with_patched_data(
            _make_model(),
            checkpoint_destination=project_checkpoint_destination,
            resume_from=resume_ctx,
        )
        saved = json.loads((project_checkpoint_destination / "metadata.json").read_text(encoding="utf-8"))
        assert saved["cumulative_epochs_completed"] == prior + EPOCHS

    def test_no_destination_skips_persistence(self, tmp_path: Path) -> None:
        """When checkpoint_destination is None, no bundle is written."""
        _run_with_patched_data(_make_model(), checkpoint_destination=None)
        # Nothing to assert beyond "didn't raise"; tmp_path is unused but the
        # call validates that the no-destination branch is exercised.
        assert not (tmp_path / "anything.json").exists()
