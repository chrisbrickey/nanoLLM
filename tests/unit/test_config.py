"""Unit tests for src/config/ dataclasses."""

import pytest

from src.config import (
    MIN_NEW_TOKENS,
    MIN_TEMPERATURE_EXCLUSIVE,
    NEUTRAL_TEMPERATURE,
    RECOMMENDED_NEW_TOKENS,
    RECOMMENDED_TEMPERATURE,
    InferenceConfig,
    ModelConfig,
    ParamRange,
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

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(delimiter=""), "delimiter"),
            (dict(pad_token_id=-1), "pad_token_id"),
            (dict(name="not-a-real-encoding-xyz"), "name"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict, match: str) -> None:
        """TokenizerConfig should raise ValueError on invalid fields, naming the offending field."""
        with pytest.raises(ValueError, match=match):
            TokenizerConfig(**kwargs)


class TestModelConfig:
    """Test suite for ModelConfig dataclass"""

    def test_config_with_valid_parameters(self) -> None:
        """Test that ModelConfig initializes with valid embed_dim and num_heads."""
        config = ModelConfig(embed_dim=192, num_heads=6)
        assert config.embed_dim == 192
        assert config.num_heads == 6

    def test_validation_fails_when_embed_dim_not_divisible_by_num_heads(self) -> None:
        """Test that __post_init__ raises ValueError for non-divisible dimensions."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
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

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(maxlen=0), "maxlen"),
            (dict(maxlen=-1), "maxlen"),
            (dict(vocab_size=0), "vocab_size"),
            (dict(embed_dim=0, num_heads=1), "embed_dim"),
            (dict(num_heads=0), "num_heads"),
            (dict(feed_forward_dim=0), "feed_forward_dim"),
            (dict(num_transformer_blocks=0), "num_transformer_blocks"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict, match: str) -> None:
        """ModelConfig should raise ValueError on non-positive integer fields, naming the offending field."""
        with pytest.raises(ValueError, match=match):
            ModelConfig(**kwargs)


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

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(batch_size=0), "batch_size"),
            (dict(batch_size=-1), "batch_size"),
            (dict(epochs=0), "epochs"),
            (dict(max_stories=0), "max_stories"),
            (dict(warmup_rate=-0.1), "warmup_rate"),
            (dict(warmup_rate=1.5), "warmup_rate"),
            (dict(lr_init_value=-1e-4), "lr_init_value"),
            (dict(lr_init_value=1.0, lr_peak_value=0.1), "lr_init_value"),
            (dict(lr_end_value=-1e-6), "lr_end_value"),
            (dict(weight_decay=-0.01), "weight_decay"),
            (dict(log_every_n_steps=0), "log_every_n_steps"),
            (dict(num_workers=-1), "num_workers"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict, match: str) -> None:
        """TrainingConfig should raise ValueError on invalid fields, naming the offending field."""
        with pytest.raises(ValueError, match=match):
            TrainingConfig(**kwargs)


class TestParamRange:
    """Test suite for the ParamRange value object"""

    def test_contains_value_inside_range(self) -> None:
        param_range = ParamRange(minimum=0.1, maximum=2.0)
        assert param_range.contains(1.0) is True

    @pytest.mark.parametrize("value", [0.1, 2.0])
    def test_contains_is_inclusive_of_boundaries(self, value: float) -> None:
        param_range = ParamRange(minimum=0.1, maximum=2.0)
        assert param_range.contains(value) is True

    @pytest.mark.parametrize("value", [0.09, 2.01])
    def test_contains_excludes_values_outside_range(self, value: float) -> None:
        param_range = ParamRange(minimum=0.1, maximum=2.0)
        assert param_range.contains(value) is False


class TestInferenceConstants:
    """Test suite for the concrete inference-domain constants"""

    def test_min_new_tokens_value(self) -> None:
        assert MIN_NEW_TOKENS == 1

    def test_min_temperature_exclusive_value(self) -> None:
        assert MIN_TEMPERATURE_EXCLUSIVE == 0.0

    def test_neutral_temperature_value(self) -> None:
        assert NEUTRAL_TEMPERATURE == 1.0

    def test_recommended_temperature_values(self) -> None:
        assert RECOMMENDED_TEMPERATURE.minimum == 0.1
        assert RECOMMENDED_TEMPERATURE.maximum == 2.0

    def test_recommended_new_tokens_values(self) -> None:
        assert RECOMMENDED_NEW_TOKENS.minimum == 1
        assert RECOMMENDED_NEW_TOKENS.maximum == 200

    def test_recommended_temperature_stays_within_hard_limits(self) -> None:
        """This prevents a regression where the recommended band is set to inappropriate values."""
        assert RECOMMENDED_TEMPERATURE.minimum > MIN_TEMPERATURE_EXCLUSIVE
        assert RECOMMENDED_TEMPERATURE.maximum > RECOMMENDED_TEMPERATURE.minimum

    def test_recommended_new_tokens_stays_within_hard_limits(self) -> None:
        """This prevents a regression where the recommended band is set to inappropriate values."""
        assert RECOMMENDED_NEW_TOKENS.minimum >= MIN_NEW_TOKENS
        assert RECOMMENDED_NEW_TOKENS.maximum > RECOMMENDED_NEW_TOKENS.minimum


class TestInferenceConfig:
    """Test suite for InferenceConfig dataclass"""

    def test_default_values(self) -> None:
        """Every inference surface starts from these defaults, so pin them here."""
        config = InferenceConfig()
        assert config.max_new_tokens == 30
        assert config.temperature == 1.0
        assert config.seed is None

    def test_default_temperature_leaves_scores_unscaled(self) -> None:
        """Temperature 1.0 divides raw scores by one, which is the model's own distribution."""
        assert InferenceConfig().temperature == NEUTRAL_TEMPERATURE

    def test_defaults_fall_inside_recommended_bands(self) -> None:
        """A default outside its own recommended band would warn on every unmodified run."""
        config = InferenceConfig()
        assert RECOMMENDED_TEMPERATURE.contains(config.temperature)
        assert RECOMMENDED_NEW_TOKENS.contains(config.max_new_tokens)

    def test_accepts_temperature_outside_recommended_range(self) -> None:
        """RECOMMENDED_TEMPERATURE is a recommendation, not a hard limit; the CLI is a research surface."""
        out_of_range_temperature = 5.0
        assert not RECOMMENDED_TEMPERATURE.contains(out_of_range_temperature)

        config = InferenceConfig(temperature=out_of_range_temperature)

        assert config.temperature == out_of_range_temperature

    def test_custom_values(self) -> None:
        max_new_tokens, temperature, seed = 10, 0.5, 42
        config = InferenceConfig(max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)

        assert config.max_new_tokens == max_new_tokens
        assert config.temperature == temperature
        assert config.seed == seed

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(max_new_tokens=0), "max_new_tokens"),
            (dict(max_new_tokens=-1), "max_new_tokens"),
            (dict(temperature=0.0), "temperature"),
            (dict(temperature=-0.5), "temperature"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict, match: str) -> None:
        """InferenceConfig should raise ValueError on invalid fields, naming the offending field."""
        with pytest.raises(ValueError, match=match):
            InferenceConfig(**kwargs)
