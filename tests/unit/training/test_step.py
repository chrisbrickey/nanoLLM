"""Unit tests for src/training/step.py"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.config import ModelConfig
from src.loss import cross_entropy_loss
from src.model.model import NanoLLM
from src.training.step import make_train_step

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1
BATCH_SIZE = 2


def _make_model() -> NanoLLM:
    config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
    )
    return NanoLLM(config)


def _make_optimizer(model: NanoLLM) -> nnx.ModelAndOptimizer:
    return nnx.ModelAndOptimizer(model, optax.adamw(learning_rate=1e-3, weight_decay=0.01))


def _make_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"))


def _make_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    inputs = jnp.ones((BATCH_SIZE, MAXLEN), dtype=jnp.int32)
    targets = jnp.ones((BATCH_SIZE, MAXLEN), dtype=jnp.int32)
    return inputs, targets


class TestMakeTrainStep:
    def test_params_update_after_one_step(self) -> None:
        model = _make_model()
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

    def test_metrics_accumulate_finite_loss(self) -> None:
        model = _make_model()
        optimizer = _make_optimizer(model)
        metrics = _make_metrics()
        batch = _make_batch()

        train_step = make_train_step()
        train_step(model, optimizer, metrics, batch)

        result = metrics.compute()
        assert jnp.isfinite(result["loss"])
        assert float(result["loss"]) > 0.0

    def test_multiple_steps_average_correctly(self) -> None:
        model = _make_model()
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

    def test_gradients_are_nonzero(self) -> None:
        model = _make_model()
        batch = _make_batch()

        grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
        _outputs, grads = grad_fn(model, batch)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0) for g in grad_leaves)
