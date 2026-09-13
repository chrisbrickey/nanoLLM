"""Shared fixtures for the inference unit tests"""

from unittest.mock import MagicMock

import pytest

from src.config import TokenizerConfig


@pytest.fixture
def mock_model() -> MagicMock:
    """A stand-in model for tests that never reach a forward pass."""
    return MagicMock()


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig()
