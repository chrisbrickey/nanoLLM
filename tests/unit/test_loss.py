"""Unit tests for src/loss.py"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.loss import cross_entropy_loss
from src.model.model import NanoLLM
from tests.conftest import TINY_MAXLEN, TINY_VOCAB_SIZE

BATCH_SIZE = 2


def _make_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    inputs = jnp.ones((BATCH_SIZE, TINY_MAXLEN), dtype=jnp.int32)
    targets = jnp.ones((BATCH_SIZE, TINY_MAXLEN), dtype=jnp.int32)
    return inputs, targets


class TestLossFn:
    def test_returns_scalar_loss_and_correct_logit_shape(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        loss, logits = cross_entropy_loss(make_tiny_model(), _make_batch())
        assert loss.shape == ()
        assert logits.shape == (BATCH_SIZE, TINY_MAXLEN, TINY_VOCAB_SIZE)

    def test_loss_is_non_negative(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        loss, _ = cross_entropy_loss(make_tiny_model(), _make_batch())
        assert float(loss) >= 0.0

    def test_loss_is_deterministic(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        model = make_tiny_model()
        batch = _make_batch()
        loss_a, _ = cross_entropy_loss(model, batch)
        loss_b, _ = cross_entropy_loss(model, batch)
        assert jnp.allclose(loss_a, loss_b)

    def test_gradients_are_nonzero(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        """cross_entropy_loss must produce nonzero gradients for at least one
        leaf — a sanity check that the loss flows backward through the model."""
        grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
        _outputs, grads = grad_fn(make_tiny_model(), _make_batch())

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0) for g in grad_leaves)
