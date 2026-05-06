from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for NanoLLM model architecture."""

    # Sequence and vocabulary
    maxlen: int = 128
    vocab_size: int = 50257  # GPT-2 tokenizer default

    # Embedding dimensions
    embed_dim: int = 192

    # Attention configuration
    num_heads: int = 6

    # Feed-forward dimension (default: 2/3 * 4 * embed_dim)
    feed_forward_dim: int = 512

    # Number of transformer blocks
    num_transformer_blocks: int = 6

    # Random seed for model initialization
    model_seed: int = 0

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Uses raise (not assert) so checks survive `python -O`.
        """
        if self.maxlen <= 0:
            raise ValueError(f"maxlen must be > 0, got {self.maxlen}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.feed_forward_dim <= 0:
            raise ValueError(f"feed_forward_dim must be > 0, got {self.feed_forward_dim}")
        if self.num_transformer_blocks <= 0:
            raise ValueError(
                f"num_transformer_blocks must be > 0, got {self.num_transformer_blocks}"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
