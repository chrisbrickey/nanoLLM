"""Unit tests for TokenAndPositionEmbedding."""

import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.model.embeddings import TokenAndPositionEmbedding

MAXLEN = 8
VOCAB_SIZE = 100
EMBED_DIM = 16
BATCH_SIZE = 2
SEQ_LEN = 4
SEED = 0


@pytest.fixture
def embedding() -> TokenAndPositionEmbedding:
    return TokenAndPositionEmbedding(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        rngs=nnx.Rngs(SEED),
    )


@pytest.fixture
def sample_token_ids() -> jnp.ndarray:
    return jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # shape (BATCH_SIZE, SEQ_LEN)


class TestTokenAndPositionEmbeddingShape:
    def test_output_shape(
        self, embedding: TokenAndPositionEmbedding, sample_token_ids: jnp.ndarray
    ) -> None:
        output = embedding(sample_token_ids)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    def test_output_dtype_is_float(
        self, embedding: TokenAndPositionEmbedding, sample_token_ids: jnp.ndarray
    ) -> None:
        output = embedding(sample_token_ids)
        assert jnp.issubdtype(output.dtype, jnp.floating)


class TestTokenAndPositionEmbeddingValues:
    def test_same_token_at_different_positions_differs(
        self, embedding: TokenAndPositionEmbedding
    ) -> None:
        # Token 1 at position 0 vs token 1 at position 1 — positional encoding distinguishes them
        tokens = jnp.array([[1, 1]])
        output = embedding(tokens)
        assert not jnp.allclose(output[0, 0, :], output[0, 1, :])

    def test_different_tokens_at_same_position_differ(
        self, embedding: TokenAndPositionEmbedding
    ) -> None:
        tokens_a = jnp.array([[1, 0]])
        tokens_b = jnp.array([[2, 0]])
        out_a = embedding(tokens_a)
        out_b = embedding(tokens_b)
        assert not jnp.allclose(out_a[0, 0, :], out_b[0, 0, :])
