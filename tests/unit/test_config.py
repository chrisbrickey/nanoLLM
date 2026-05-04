"""Unit tests for src/config/ dataclasses."""

import pytest

from src.config import (
    InferenceConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)


class TestTokenizerConfig:
    """Test suite for TokenizerConfig dataclass"""

    def test_default_values(self) -> None:
        """Defaults target gpt2 with the standard end-of-text marker."""
        config = TokenizerConfig()
        assert config.delimiter == "<|endoftext|>"
        assert config.name == "gpt2"
        assert config.pad_token_id == 0

    def test_config_initialization(self) -> None:
        """Test that TokenizerConfig can be initialized with custom fields."""
        config = TokenizerConfig(delimiter="<|test|>", name="gpt2", pad_token_id=1)
        assert config.delimiter == "<|test|>"
        assert config.name == "gpt2"
        assert config.pad_token_id == 1

    def test_tokenizer_property_returns_encoding(self) -> None:
        """Test that tokenizer property returns a tiktoken Encoding instance."""
        import tiktoken

        config = TokenizerConfig(delimiter="<|test|>", name="gpt2")
        tokenizer = config.tokenizer
        assert isinstance(tokenizer, tiktoken.Encoding)
        assert tokenizer.name == "gpt2"

    def test_vocab_size_property(self) -> None:
        """Test that vocab_size property returns correct vocabulary size."""
        config = TokenizerConfig(delimiter="<|test|>", name="gpt2")
        vocab_size = config.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0

    def test_end_token_aliases_delimiter(self) -> None:
        """Test that end_token property returns the delimiter value."""
        test_delimiter = "<|custom|>"
        config = TokenizerConfig(delimiter=test_delimiter, name="gpt2")
        assert config.end_token == test_delimiter


class TestModelConfig:
    """Test suite for ModelConfig dataclass"""

    def test_config_with_valid_parameters(self) -> None:
        """Test that ModelConfig initializes with valid embed_dim and num_heads."""
        config = ModelConfig(embed_dim=192, num_heads=6)
        assert config.embed_dim == 192
        assert config.num_heads == 6

    def test_validation_fails_when_embed_dim_not_divisible_by_num_heads(self) -> None:
        """Test that __post_init__ raises AssertionError for invalid dimensions."""
        with pytest.raises(AssertionError, match="embed_dim .* must be divisible by num_heads"):
            ModelConfig(embed_dim=100, num_heads=7)

    def test_default_values(self) -> None:
        """Test that ModelConfig has sensible default values."""
        config = ModelConfig()
        assert config.maxlen > 0
        assert config.vocab_size > 0
        assert config.embed_dim > 0
        assert config.num_heads > 0
        assert config.num_transformer_blocks > 0

    def test_no_module_level_singleton_exported(self) -> None:
        """ModelConfig is checkpoint-bound; no global singleton should be exported."""
        import src.config as config_module

        assert not hasattr(config_module, "model_config")


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass"""

    def test_default_values(self) -> None:
        """Test that TrainingConfig has sensible default values."""
        config = TrainingConfig()
        assert config.batch_size > 0
        assert config.epochs > 0
        assert 0.0 <= config.warmup_rate <= 1.0
        assert config.lr_peak_value > config.lr_init_value
        assert config.lr_peak_value > config.lr_end_value


class TestInferenceConfig:
    """Test suite for InferenceConfig dataclass"""

    def test_default_values(self) -> None:
        """Test that InferenceConfig has sensible default values."""
        config = InferenceConfig()
        assert config.max_new_tokens > 0
        assert config.temperature > 0.0

    def test_custom_values(self) -> None:
        config = InferenceConfig(max_new_tokens=10, temperature=0.5)
        assert config.max_new_tokens == 10
        assert config.temperature == 0.5
