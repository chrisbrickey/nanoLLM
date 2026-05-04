"""Unit tests for src/model/model.py"""

import jax.numpy as jnp
import pytest

from src.config import ModelConfig
from src.model.model import NanoLLM

# Test constants - generic sequence lengths
SEQ_LENGTH_SMALL = 4
SEQ_LENGTH_MEDIUM = 8
SEQ_LENGTH_LARGE = 16

# Small model dimensions for fast forward-pass tests
SMALL_MAXLEN = 4
SMALL_VOCAB_SIZE = 200
SMALL_EMBED_DIM = 12  # divisible by SMALL_NUM_HEADS
SMALL_NUM_HEADS = 3
SMALL_FF_DIM = 16
SMALL_NUM_BLOCKS = 1


def _small_config() -> ModelConfig:
    return ModelConfig(
        maxlen=SMALL_MAXLEN,
        vocab_size=SMALL_VOCAB_SIZE,
        embed_dim=SMALL_EMBED_DIM,
        num_heads=SMALL_NUM_HEADS,
        feed_forward_dim=SMALL_FF_DIM,
        num_transformer_blocks=SMALL_NUM_BLOCKS,
    )


@pytest.fixture
def small_model() -> NanoLLM:
    return NanoLLM(_small_config())


class TestCausalAttentionMask:
    """Test suite for NanoLLM.causal_attention_mask() method"""

    def test_returns_lower_triangular_matrix(self, small_model: NanoLLM) -> None:
        """Test that mask is lower triangular (1s below/on diagonal, 0s above)"""
        mask = small_model.causal_attention_mask(SEQ_LENGTH_SMALL)

        # Verify it's lower triangular by checking upper triangle is all zeros
        for i in range(SEQ_LENGTH_SMALL):
            for j in range(SEQ_LENGTH_SMALL):
                if j > i:  # Upper triangle
                    assert mask[i, j] == 0, f"Expected 0 at position ({i}, {j}), got {mask[i, j]}"
                else:  # Lower triangle + diagonal
                    assert mask[i, j] == 1, f"Expected 1 at position ({i}, {j}), got {mask[i, j]}"

    def test_correct_shape_for_various_sequence_lengths(self, small_model: NanoLLM) -> None:
        """Test that mask shape matches (seq_len, seq_len) for different lengths"""
        test_lengths = [SEQ_LENGTH_SMALL, SEQ_LENGTH_MEDIUM, SEQ_LENGTH_LARGE]

        for seq_len in test_lengths:
            mask = small_model.causal_attention_mask(seq_len)
            assert mask.shape == (seq_len, seq_len), \
                f"Expected shape ({seq_len}, {seq_len}), got {mask.shape}"

    def test_mask_values_are_binary(self, small_model: NanoLLM) -> None:
        """Test that mask contains only 0s and 1s"""
        mask = small_model.causal_attention_mask(SEQ_LENGTH_MEDIUM)

        # Check all values are either 0 or 1
        unique_values = jnp.unique(mask)
        assert len(unique_values) <= 2, f"Expected only 0s and 1s, got {unique_values}"
        assert all(val in [0, 1] for val in unique_values), \
            f"Expected only 0s and 1s, got {unique_values}"

    def test_diagonal_elements_are_ones(self, small_model: NanoLLM) -> None:
        """Test that diagonal elements are 1 (tokens can attend to themselves)"""
        mask = small_model.causal_attention_mask(SEQ_LENGTH_SMALL)

        # Check diagonal is all 1s
        diagonal = jnp.diag(mask)
        assert jnp.all(diagonal == 1), \
            f"Expected all diagonal elements to be 1, got {diagonal}"


class TestNanoLLMForwardPass:
    """Test suite for NanoLLM.__call__()"""

    def test_output_shape(self, small_model: NanoLLM) -> None:
        batch_size = 2
        token_ids = jnp.zeros((batch_size, SMALL_MAXLEN), dtype=jnp.int32)
        logits = small_model(token_ids)
        assert logits.shape == (batch_size, SMALL_MAXLEN, SMALL_VOCAB_SIZE)

    def test_output_dtype_is_float(self, small_model: NanoLLM) -> None:
        token_ids = jnp.zeros((1, SMALL_MAXLEN), dtype=jnp.int32)
        logits = small_model(token_ids)
        assert jnp.issubdtype(logits.dtype, jnp.floating)

    def test_different_inputs_give_different_logits(self, small_model: NanoLLM) -> None:
        ids_a = jnp.zeros((1, SMALL_MAXLEN), dtype=jnp.int32)
        ids_b = jnp.ones((1, SMALL_MAXLEN), dtype=jnp.int32)
        assert not jnp.allclose(small_model(ids_a), small_model(ids_b))
