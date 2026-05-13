"""NanoLLM transformer model architecture."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.config import ModelConfig
from src.model.embeddings import TokenAndPositionEmbedding
from src.model.blocks import TransformerBlock


def count_params(model: nnx.Module) -> int:
    """Return total number of trainable parameters in the model.

    This is a standalone utility because it can work on an instance
    of NanoLLM but it can also work for any flax model."""
    params = nnx.state(model, nnx.Param)
    return sum(v.size for v in jax.tree_util.tree_leaves(params))


class NanoLLM(nnx.Module):
    """
    A nano-scale transformer language model.

    This model combines token/position embeddings, stacked transformer blocks,
    and an output layer to predict next tokens in a sequence.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize NanoLLM model.

        Args:
            config: Architecture configuration. For inference, this should be
                reconstructed from checkpoint metadata so weight shapes match.
                Set config.model_seed to control weight initialization.
        """
        rngs = nnx.Rngs(config.model_seed)
        self.maxlen = config.maxlen
        self.config = config

        # create token and position vector embeddings
        self.embedding = TokenAndPositionEmbedding(
            config.maxlen, config.vocab_size, config.embed_dim, rngs=rngs
        )

        # applies num_transformer_blocks of transformer layers sequentially
        self.transformer_blocks = nnx.List([
            TransformerBlock(
                config.embed_dim, config.num_heads, config.feed_forward_dim, rngs=rngs
            )
            for _ in range(config.num_transformer_blocks)
        ])

        # output of dense vectors (hidden states) that capture a position's meaning
        self.output_layer = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def causal_attention_mask(self, seq_len: int) -> jnp.ndarray:
        """
        Prevents the LLM from paying attention to words ahead of the current position
        so that the LLM cannot see the next word in the phrase before it makes a prediction.

        Blocks the LLM from accessing embeddings in the table where position > current position.

        Args: seq_len: Sequence length

        Returns: Lower triangular mask of shape (seq_len, seq_len)
        """
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """

        # Python-side static-shape check (not a traced JAX op) to surface a clear
        # error before the position-embedding lookup raises an opaque index error.
        seq_len = token_ids.shape[1]
        if seq_len > self.maxlen:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds model maxlen ({self.maxlen})"
            )

        x = self.embedding(token_ids)

        mask = self.causal_attention_mask(seq_len)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        # raw scores for next-token prediction (then choose the highest score)
        logits = self.output_layer(x)
        return logits
