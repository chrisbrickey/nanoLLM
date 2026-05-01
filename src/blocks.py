"""Transformer block components for the model architecture."""

import jax.numpy as jnp
import flax.nnx as nnx


class TransformerBlock(nnx.Module):
    """
    A single transformer block with multi-head attention.

    This block applies self-attention followed by residual connection.
    Future iterations may add layer normalization and feed-forward layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs
    ) -> None:
        """
        Initialize transformer block.

        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward layer (currently unused)
            rngs: Random number generator for initialization
        """
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """
        Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # skip some heads sometimes which generally adds stability to the model ("residual connections")
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        return x
