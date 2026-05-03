"""Unit tests for src/training/trainer.py"""

import logging
import shutil
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import flax.nnx as nnx
import pytest

from src.config import CHECKPOINTS_DIR, TrainingConfig
from src.model.model import NanoLLM
from src.training.trainer import Trainer

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1
BATCH_SIZE = 2
N_BATCHES = 4  # yields 2 log entries with default log_every_n_steps=2
SEED = 0


class _FakeDataLoader:
    """Mimics grain's batch format: each batch is shape (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


def _make_model() -> NanoLLM:
    return NanoLLM(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        rngs=nnx.Rngs(SEED),
    )


def _make_config(**overrides: object) -> TrainingConfig:
    defaults = dict(
        num_epochs=1,
        batch_size=BATCH_SIZE,
        log_every_n_steps=2,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _make_dataloader() -> _FakeDataLoader:
    return _FakeDataLoader(n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE)


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"trainer_test_{uuid.uuid4().hex[:8]}.orbax"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestTrainerInit:
    def test_wires_optimizer_and_metrics_and_step(self) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
        )
        assert isinstance(trainer.optimizer, nnx.ModelAndOptimizer)
        assert isinstance(trainer.metrics, nnx.MultiMetric)
        assert callable(trainer.train_step)


class TestTrainerTrain:
    def test_returns_populated_metrics_history(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert "loss=" in caplog.text
        assert "Epoch 1/1 complete" in caplog.text

    def test_log_every_n_steps_controls_history_length(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _make_config(log_every_n_steps=2)
        trainer = Trainer(
            model=_make_model(),
            training_config=config,
            dataloader=_make_dataloader(),  # N_BATCHES=4 batches
            batches_per_epoch=N_BATCHES,
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        # 4 batches / log every 2 steps = 2 entries
        assert len(history["train_loss"]) == N_BATCHES // config.log_every_n_steps
        assert caplog.text.count("loss=") == N_BATCHES // config.log_every_n_steps

    def test_empty_dataloader_returns_empty_history(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_FakeDataLoader(n_batches=0, maxlen=MAXLEN, batch_size=BATCH_SIZE),
            batches_per_epoch=10,  # enough for a valid schedule; dataloader yields nothing
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert history == {"train_loss": []}
        assert "loss=" not in caplog.text

    def test_checkpoint_path_none_skips_save(self) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=None,
        )
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            mock_save.assert_not_called()

    def test_checkpoint_path_writes_file(self, project_checkpoint_path: Path) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=project_checkpoint_path,
        )
        trainer.train()
        assert project_checkpoint_path.exists()
