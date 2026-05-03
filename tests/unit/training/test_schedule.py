"""Unit tests for src/training/schedule.py"""

import jax.numpy as jnp
import pytest

from src.config import TrainingConfig
from src.training.schedule import build_learning_rate_schedule

TOTAL_STEPS = 100
WARMUP_STEPS = 10
LR_INIT = 0.0
LR_PEAK = 1e-2
LR_END = 1e-5


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        lr_init_value=LR_INIT,
        lr_peak_value=LR_PEAK,
        lr_end_value=LR_END,
    )


class TestBuildLearningRateSchedule:
    def test_starts_at_init_value(self, config: TrainingConfig) -> None:
        schedule = build_learning_rate_schedule(config, TOTAL_STEPS, WARMUP_STEPS)
        assert jnp.allclose(schedule(0), jnp.array(LR_INIT))

    def test_reaches_peak_at_end_of_warmup(self, config: TrainingConfig) -> None:
        schedule = build_learning_rate_schedule(config, TOTAL_STEPS, WARMUP_STEPS)
        assert jnp.allclose(schedule(WARMUP_STEPS), jnp.array(LR_PEAK))

    def test_decays_to_end_value_at_total_steps(self, config: TrainingConfig) -> None:
        schedule = build_learning_rate_schedule(config, TOTAL_STEPS, WARMUP_STEPS)
        assert jnp.allclose(schedule(TOTAL_STEPS), jnp.array(LR_END), atol=1e-7)

    def test_does_not_drop_below_end_value_past_total_steps(self, config: TrainingConfig) -> None:
        schedule = build_learning_rate_schedule(config, TOTAL_STEPS, WARMUP_STEPS)
        assert float(schedule(TOTAL_STEPS * 2)) >= LR_END

    def test_monotonically_decreases_during_decay(self, config: TrainingConfig) -> None:
        schedule = build_learning_rate_schedule(config, TOTAL_STEPS, WARMUP_STEPS)
        decay_steps = range(WARMUP_STEPS, TOTAL_STEPS + 1)
        values = [float(schedule(s)) for s in decay_steps]
        assert all(a >= b for a, b in zip(values, values[1:]))
