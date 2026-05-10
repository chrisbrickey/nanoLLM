"""Shared fixtures for unit and integration tests.

Centralizes the tiny model dimensions used across loss, step, trainer,
checkpoint, and compare test modules so each test module doesn't have to
re-declare the same constants and factory helpers.
"""

from collections.abc import Callable

import pytest

from src.config import ModelConfig
from src.model.model import NanoLLM

TINY_MAXLEN = 4
TINY_VOCAB_SIZE = 50
TINY_EMBED_DIM = 12
TINY_NUM_HEADS = 3
TINY_FF_DIM = 16
TINY_NUM_BLOCKS = 1


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        maxlen=TINY_MAXLEN,
        vocab_size=TINY_VOCAB_SIZE,
        embed_dim=TINY_EMBED_DIM,
        num_heads=TINY_NUM_HEADS,
        feed_forward_dim=TINY_FF_DIM,
        num_transformer_blocks=TINY_NUM_BLOCKS,
    )


@pytest.fixture
def make_tiny_model() -> Callable[..., NanoLLM]:
    def _make(seed: int = 0) -> NanoLLM:
        return NanoLLM(
            ModelConfig(
                maxlen=TINY_MAXLEN,
                vocab_size=TINY_VOCAB_SIZE,
                embed_dim=TINY_EMBED_DIM,
                num_heads=TINY_NUM_HEADS,
                feed_forward_dim=TINY_FF_DIM,
                num_transformer_blocks=TINY_NUM_BLOCKS,
                model_seed=seed,
            )
        )

    return _make
