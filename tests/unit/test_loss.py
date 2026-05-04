"""Unit tests for src/loss.py"""

import jax.numpy as jnp

from src.config import ModelConfig
from src.loss import cross_entropy_loss
from src.model.model import NanoLLM

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


def _make_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    inputs = jnp.ones((BATCH_SIZE, MAXLEN), dtype=jnp.int32)
    targets = jnp.ones((BATCH_SIZE, MAXLEN), dtype=jnp.int32)
    return inputs, targets


class TestLossFn:
    def test_returns_scalar_loss_and_correct_logit_shape(self) -> None:
        model = _make_model()
        batch = _make_batch()
        loss, logits = cross_entropy_loss(model, batch)
        assert loss.shape == ()
        assert logits.shape == (BATCH_SIZE, MAXLEN, VOCAB_SIZE)

    def test_loss_is_non_negative(self) -> None:
        model = _make_model()
        batch = _make_batch()
        loss, _ = cross_entropy_loss(model, batch)
        assert float(loss) >= 0.0

    def test_loss_is_deterministic(self) -> None:
        model = _make_model()
        batch = _make_batch()
        loss_a, _ = cross_entropy_loss(model, batch)
        loss_b, _ = cross_entropy_loss(model, batch)
        assert jnp.allclose(loss_a, loss_b)
