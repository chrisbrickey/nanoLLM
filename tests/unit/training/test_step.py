"""Unit tests for src/training/step.py"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.model.model import NanoLLM
from src.training.step import make_train_step
from tests.conftest import TINY_MAXLEN

BATCH_SIZE = 2


def _make_optimizer(model: NanoLLM) -> nnx.ModelAndOptimizer:
    return nnx.ModelAndOptimizer(model, optax.adamw(learning_rate=1e-3, weight_decay=0.01))


def _make_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"))


def _make_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    inputs = jnp.ones((BATCH_SIZE, TINY_MAXLEN), dtype=jnp.int32)
    targets = jnp.ones((BATCH_SIZE, TINY_MAXLEN), dtype=jnp.int32)
    return inputs, targets


class TestMakeTrainStep:
    def test_params_update_after_one_step(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        model = make_tiny_model()
        optimizer = _make_optimizer(model)
        metrics = _make_metrics()
        batch = _make_batch()

        params_before = jax.tree_util.tree_leaves(nnx.state(model))

        train_step = make_train_step()
        train_step(model, optimizer, metrics, batch)

        params_after = jax.tree_util.tree_leaves(nnx.state(model))
        assert not all(
            jnp.allclose(b, a) for b, a in zip(params_before, params_after)
        )

    def test_metrics_accumulate_finite_loss(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        model = make_tiny_model()
        optimizer = _make_optimizer(model)
        metrics = _make_metrics()
        batch = _make_batch()

        train_step = make_train_step()
        train_step(model, optimizer, metrics, batch)

        result = metrics.compute()
        assert jnp.isfinite(result["loss"])
        assert float(result["loss"]) > 0.0

    def test_multiple_steps_average_correctly(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        model = make_tiny_model()
        # lr=0 freezes weights so every step sees identical loss — average must equal any single step
        frozen_optimizer = nnx.ModelAndOptimizer(model, optax.sgd(learning_rate=0.0))
        accumulated_metrics = _make_metrics()
        batch = _make_batch()

        train_step = make_train_step()
        for _ in range(3):
            train_step(model, frozen_optimizer, accumulated_metrics, batch)

        single_metrics = _make_metrics()
        train_step(model, frozen_optimizer, single_metrics, batch)

        assert jnp.allclose(
            accumulated_metrics.compute()["loss"],
            single_metrics.compute()["loss"],
            atol=1e-5,
        )
