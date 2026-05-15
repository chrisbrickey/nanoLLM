"""Unit tests for src/training/trainer.py — Trainer.train_step is stubbed
so each test exercises the orchestration loop only (no JIT compile, no real
gradient computation, no disk I/O). Real-training behavior and disk side
effects are covered by tests/integration/training/."""

import logging
from collections.abc import Callable, Iterator

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import pytest

from src.config import TrainingConfig
from src.model.model import NanoLLM
from src.training.trainer import Trainer

EPOCH_COUNT = 1
BATCH_SIZE = 2
N_BATCHES = 4  # yields 2 log entries with default log_every_n_steps=2
MAXLEN = 4  # matches conftest.TINY_MAXLEN
STUB_LOSS = 0.5


class _FakeDataLoader:
    """Mimics grain's batch format: each batch is shape (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


def _make_config(**overrides: object) -> TrainingConfig:
    defaults = dict(epochs=EPOCH_COUNT, batch_size=BATCH_SIZE, log_every_n_steps=2)
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _make_dataloader(n_batches: int = N_BATCHES) -> _FakeDataLoader:
    return _FakeDataLoader(n_batches=n_batches, maxlen=MAXLEN, batch_size=BATCH_SIZE)


def _stub_train_step(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, jnp.ndarray],
) -> None:
    """Lightweight replacement for the JIT-compiled train_step. Updates
    metrics with a fixed loss so the orchestration loop still has a value
    to log and accumulate, but skips the gradient computation."""
    metrics.update(loss=jnp.array(STUB_LOSS))


def _build_trainer(
    model: NanoLLM, *, training_config: TrainingConfig | None = None, dataloader: _FakeDataLoader | None = None
) -> Trainer:
    trainer = Trainer(
        model=model,
        training_config=training_config or _make_config(),
        dataloader=dataloader or _make_dataloader(),
        batches_per_epoch=N_BATCHES,
    )
    trainer.train_step = _stub_train_step  # type: ignore[assignment]
    return trainer


class TestTrainerInit:
    def test_wires_optimizer_and_metrics_and_step(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
        )
        assert isinstance(trainer.optimizer, nnx.ModelAndOptimizer)
        assert isinstance(trainer.metrics, nnx.MultiMetric)
        assert callable(trainer.train_step)


class TestTrainerTrain:
    def test_train_populates_metrics_history_and_logs_progress(
        self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _make_config()
        trainer = _build_trainer(make_tiny_model(), training_config=config)
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()

        # Assert metrics history is returned as expected
        assert history["train_loss"] == [STUB_LOSS] * (N_BATCHES // config.log_every_n_steps)

        # Assert progress is logged as expected
        assert "Epoch 1 commenced" in caplog.text
        assert f"All {EPOCH_COUNT} epochs completed" in caplog.text
        assert "Loss=" in caplog.text

    def test_log_every_n_steps_controls_history_length(
        self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _make_config(log_every_n_steps=2)
        trainer = _build_trainer(make_tiny_model(), training_config=config)
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        # 4 batches / log every 2 steps = 2 entries
        assert len(history["train_loss"]) == N_BATCHES // config.log_every_n_steps
        assert caplog.text.count("Loss=") == N_BATCHES // config.log_every_n_steps

    def test_empty_dataloader_returns_empty_history(
        self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=_make_config(),
            dataloader=_FakeDataLoader(n_batches=0, maxlen=MAXLEN, batch_size=BATCH_SIZE),
            batches_per_epoch=10,  # enough for a valid schedule; dataloader yields nothing
        )
        trainer.train_step = _stub_train_step  # type: ignore[assignment]
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert history == {"train_loss": []}
        assert "loss=" not in caplog.text
