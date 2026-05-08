"""Integration tests for a full training run (src/training/)."""

import shutil
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import pytest

from src.config import ModelConfig, TrainingConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR
from src.training.trainer import Trainer

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
def project_checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"integration_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


def _make_trainer(checkpoint_path: Path | None = None) -> Trainer:
    model_config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
    )
    model = NanoLLM(model_config)
    config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        log_every_n_steps=4,
    )
    dataloader = _FakeDataLoader(
        n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE
    )
    return Trainer(
        model=model,
        training_config=config,
        dataloader=dataloader,
        batches_per_epoch=N_BATCHES,
        checkpoint_path=checkpoint_path,
    )


class TestTrainLoop:
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

    def test_checkpoint_written_to_disk(self, project_checkpoint_path: Path) -> None:
        _make_trainer(checkpoint_path=project_checkpoint_path).train()
        assert project_checkpoint_path.exists()
        assert (project_checkpoint_path / "weights.orbax").exists()
        assert (project_checkpoint_path / "metadata.json").exists()
