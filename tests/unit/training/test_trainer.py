"""Unit tests for src/training/trainer.py.

Orchestration-only tests stub trainer.train_step so each test exercises the
loop without JIT compile or real gradient computation. Tests that need to
exercise the real JIT-compiled train_step build a Trainer and call it
directly via the public trainer.train() entry point.

Real-training behavior and disk side effects are covered by
tests/integration/training/.
"""

import logging
from collections.abc import Callable, Iterator

import jax
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

LR_INIT = 0.0
LR_PEAK = 1e-2
LR_END = 1e-5
SCHEDULE_TOTAL_STEPS = 100
SCHEDULE_WARMUP_STEPS = 10


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
    model: NanoLLM,
    *,
    training_config: TrainingConfig | None = None,
    dataloader: _FakeDataLoader | None = None,
    batches_per_epoch: int = N_BATCHES,
) -> Trainer:
    trainer = Trainer(
        model=model,
        training_config=training_config or _make_config(),
        dataloader=dataloader or _make_dataloader(),
        batches_per_epoch=batches_per_epoch,
    )
    trainer.train_step = _stub_train_step  # type: ignore[assignment]
    return trainer


def _schedule_config() -> TrainingConfig:
    return TrainingConfig(
        lr_init_value=LR_INIT,
        lr_peak_value=LR_PEAK,
        lr_end_value=LR_END,
    )


def _build_trainer_with_schedule(
    model: NanoLLM,
    *,
    config: TrainingConfig | None = None,
    batches_per_epoch: int = SCHEDULE_TOTAL_STEPS,
) -> Trainer:
    """Build a Trainer wired so total_steps/warmup_steps match the constants
    used by the schedule-shape assertions. Uses epochs=1 so
    batches_per_epoch == total_steps; warmup_rate is set so warmup_steps
    lands on SCHEDULE_WARMUP_STEPS."""
    cfg = config or TrainingConfig(
        epochs=1,
        warmup_rate=SCHEDULE_WARMUP_STEPS / SCHEDULE_TOTAL_STEPS,
        lr_init_value=LR_INIT,
        lr_peak_value=LR_PEAK,
        lr_end_value=LR_END,
    )
    return Trainer(
        model=model,
        training_config=cfg,
        dataloader=_make_dataloader(n_batches=0),
        batches_per_epoch=batches_per_epoch,
    )


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

    def test_exposes_total_and_warmup_steps(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        config = TrainingConfig(epochs=3, warmup_rate=0.1)
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=config,
            dataloader=_make_dataloader(),
            batches_per_epoch=100,
        )
        assert trainer.total_steps == 300
        assert trainer.warmup_steps == 30

    def test_warmup_floor_is_one_on_tiny_dataset(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        """warmup_steps must be at least 1 even when total_steps * warmup_rate rounds to 0."""
        config = TrainingConfig(epochs=1, warmup_rate=0.1)
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=config,
            dataloader=_make_dataloader(),
            batches_per_epoch=5,
        )
        assert trainer.total_steps == 5
        assert trainer.warmup_steps >= 1

    @pytest.mark.parametrize("batches_per_epoch", [0, -1])
    def test_rejects_non_positive_batches_per_epoch(
        self, make_tiny_model: Callable[..., NanoLLM], batches_per_epoch: int
    ) -> None:
        """Trainer must fail fast rather than produce a degenerate zero-step schedule."""
        with pytest.raises(ValueError, match="batches_per_epoch"):
            Trainer(
                model=make_tiny_model(),
                training_config=_make_config(),
                dataloader=_make_dataloader(),
                batches_per_epoch=batches_per_epoch,
            )


class TestTrainerSchedule:
    def test_starts_at_init_value(self, make_tiny_model: Callable[..., NanoLLM]) -> None:
        trainer = _build_trainer_with_schedule(make_tiny_model())
        assert jnp.allclose(trainer.schedule(0), jnp.array(LR_INIT))

    def test_reaches_peak_at_end_of_warmup(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer_with_schedule(make_tiny_model())
        assert jnp.allclose(trainer.schedule(SCHEDULE_WARMUP_STEPS), jnp.array(LR_PEAK))

    def test_decays_to_end_value_at_total_steps(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer_with_schedule(make_tiny_model())
        assert jnp.allclose(
            trainer.schedule(SCHEDULE_TOTAL_STEPS), jnp.array(LR_END), atol=1e-7
        )

    def test_does_not_drop_below_end_value_past_total_steps(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer_with_schedule(make_tiny_model())
        assert float(trainer.schedule(SCHEDULE_TOTAL_STEPS * 2)) >= LR_END

    def test_monotonically_decreases_during_decay(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer_with_schedule(make_tiny_model())
        decay_steps = range(SCHEDULE_WARMUP_STEPS, SCHEDULE_TOTAL_STEPS + 1)
        values = [float(trainer.schedule(s)) for s in decay_steps]
        assert all(a >= b for a, b in zip(values, values[1:]))


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
        assert "Training plan:" in caplog.text
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


class TestTrainerRealStep:
    """Exercises the real JIT-compiled train_step (no stub) end-to-end through
    trainer.train(). Covers what the old test_step.py guaranteed."""

    def test_params_update_after_training(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        model = make_tiny_model()
        params_before = jax.tree_util.tree_leaves(nnx.state(model))
        trainer = Trainer(
            model=model,
            training_config=_make_config(log_every_n_steps=1),
            dataloader=_make_dataloader(n_batches=2),
            batches_per_epoch=2,
        )
        trainer.train()
        params_after = jax.tree_util.tree_leaves(nnx.state(model))
        assert not all(
            jnp.allclose(b, a) for b, a in zip(params_before, params_after)
        )

    def test_metrics_history_contains_finite_positive_loss(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=_make_config(log_every_n_steps=2),
            dataloader=_make_dataloader(n_batches=2),
            batches_per_epoch=2,
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 1
        loss = history["train_loss"][0]
        assert jnp.isfinite(jnp.array(loss))
        assert loss > 0.0

    def test_frozen_weights_yield_identical_losses_across_steps(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        """With learning rate pinned to 0 the weights cannot move, so every
        batch (which is identical from the fake dataloader) must produce the
        same loss. This is the orchestration-level analog of the old
        test_multiple_steps_average_correctly."""
        frozen_config = TrainingConfig(
            epochs=1,
            batch_size=BATCH_SIZE,
            log_every_n_steps=1,
            lr_init_value=0.0,
            lr_peak_value=0.0,
            lr_end_value=0.0,
        )
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=frozen_config,
            dataloader=_make_dataloader(n_batches=3),
            batches_per_epoch=3,
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 3
        first = history["train_loss"][0]
        for loss in history["train_loss"][1:]:
            assert jnp.allclose(jnp.array(loss), jnp.array(first), atol=1e-5)
