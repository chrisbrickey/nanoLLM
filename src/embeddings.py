"""Token and position embedding layer for transformer models."""

import jax.numpy as jnp
import flax.nnx as nnx


class TokenAndPositionEmbedding(nnx.Module):
    """
    Combines token embeddings with positional embeddings.

    This layer maps input token IDs to dense vectors and adds positional
    information so the model can understand sequence order.
    """

    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs
    ) -> None:
        """
        Initialize token and position embeddings.

        Args:
            maxlen: Maximum sequence length
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embedding vectors
            rngs: Random number generator for initialization
        """

        # Text: creates table (vocab_size x embed_dim) for token vectors
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)

        # Position: creates table (maxlen x embed_dim) for position vectors (indexed 0 to maxlen-1)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply token and position embeddings.

        Args:
            x: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embedded vectors of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]

        # Combines token vectors and position vectors in an element-wise sum,
        # which injects both semantic and positional information
        return self.token_emb(x) + self.pos_emb(positions)
