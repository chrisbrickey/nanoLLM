"""Unit tests for TransformerBlock."""

import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.model.blocks import TransformerBlock

EMBED_DIM = 8
NUM_HEADS = 2
FF_DIM = 16
BATCH_SIZE = 2
SEQ_LEN = 4
SEED = 0


@pytest.fixture
def block() -> TransformerBlock:
    return TransformerBlock(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        rngs=nnx.Rngs(SEED),
    )


@pytest.fixture
def sample_input() -> jnp.ndarray:
    return jnp.ones((BATCH_SIZE, SEQ_LEN, EMBED_DIM))


@pytest.fixture
def causal_mask() -> jnp.ndarray:
    return jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN)))


class TestTransformerBlockShape:
    def test_output_shape_without_mask(
        self, block: TransformerBlock, sample_input: jnp.ndarray
    ) -> None:
        output = block(sample_input, mask=None)
        assert output.shape == sample_input.shape

    def test_output_shape_with_causal_mask(
        self, block: TransformerBlock, sample_input: jnp.ndarray, causal_mask: jnp.ndarray
    ) -> None:
        output = block(sample_input, mask=causal_mask)
        assert output.shape == sample_input.shape


class TestTransformerBlockResidual:
    def test_output_differs_from_input(
        self, block: TransformerBlock, sample_input: jnp.ndarray
    ) -> None:
        output = block(sample_input)
        assert not jnp.allclose(output, sample_input)
