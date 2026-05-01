"""NanoLLM transformer model architecture."""

import jax.numpy as jnp
import flax.nnx as nnx

from src.config import model_config
from src.embeddings import TokenAndPositionEmbedding
from src.blocks import TransformerBlock


class NanoLLM(nnx.Module):
    """
    A nano-scale transformer language model.

    This model combines token/position embeddings, stacked transformer blocks,
    and an output layer to predict next tokens in a sequence.
    """

    def __init__(
        self,
        maxlen: int = model_config.maxlen,
        vocab_size: int = model_config.vocab_size,
        embed_dim: int = model_config.embed_dim,
        num_heads: int = model_config.num_heads,
        feed_forward_dim: int = model_config.feed_forward_dim,
        num_transformer_blocks: int = model_config.num_transformer_blocks,
        *,
        rngs: nnx.Rngs = nnx.Rngs(model_config.model_seed)
    ) -> None:
        """
        Initialize NanoLLM model.

        Args:
            maxlen: Maximum sequence length
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads per block
            feed_forward_dim: Dimension of feed-forward layer
            num_transformer_blocks: Number of transformer blocks to stack
            rngs: Random number generator for initialization
        """
        self.maxlen = maxlen

        # create token and position vector embeddings
        self.embedding = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )

        # applies num_transformer_blocks of transformer layers sequentially
        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ])

        # output of dense vectors (hidden states) that capture a position's meaning
        self.output_layer = nnx.Linear(
            embed_dim, vocab_size, use_bias=False, rngs=rngs
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

        x = self.embedding(token_ids)

        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        # raw scores for next-token prediction (then choose the highest score)
        logits = self.output_layer(x)
        return logits
