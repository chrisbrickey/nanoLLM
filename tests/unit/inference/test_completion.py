"""Unit tests for completion.py

These tests verify prompt encoding and pass-through of the generated result."""


from unittest.mock import MagicMock, patch

import pytest

from src.config import InferenceConfig, TokenizerConfig
from src.inference.completion import complete_prompt

SAMPLE_PROMPT = "sample-text"
GENERATED_TEXT = "sample-text and then more words"


@pytest.fixture
def mock_model() -> MagicMock:
    return MagicMock()


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig()


@pytest.fixture
def inference_config() -> InferenceConfig:
    return InferenceConfig(max_new_tokens=5, seed=0)


class TestCompletePromptHappyPath:
    def test_returns_generated_text(
        self,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
        inference_config: InferenceConfig,
    ) -> None:
        with patch(
            "src.inference.completion.generate_text", return_value=GENERATED_TEXT
        ) as mock_generate:
            result = complete_prompt(
                model=mock_model,
                tokenizer_config=tokenizer_config,
                inference_config=inference_config,
                prompt=SAMPLE_PROMPT,
            )

        assert result == GENERATED_TEXT
        mock_generate.assert_called_once()

    def test_encodes_prompt_and_passes_as_start_tokens(
        self,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
        inference_config: InferenceConfig,
    ) -> None:
        expected_tokens = tokenizer_config.tokenizer.encode(SAMPLE_PROMPT)

        with patch(
            "src.inference.completion.generate_text", return_value=GENERATED_TEXT
        ) as mock_generate:
            complete_prompt(
                model=mock_model,
                tokenizer_config=tokenizer_config,
                inference_config=inference_config,
                prompt=SAMPLE_PROMPT,
            )

        _, kwargs = mock_generate.call_args
        assert kwargs["model"] is mock_model
        assert kwargs["tokenizer_config"] is tokenizer_config
        assert kwargs["inference_config"] is inference_config
        assert kwargs["start_tokens"] == expected_tokens


class TestCompletePromptValidation:
    def test_empty_prompt_raises_value_error(
        self,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
        inference_config: InferenceConfig,
    ) -> None:
        with pytest.raises(ValueError, match="prompt"):
            complete_prompt(
                model=mock_model,
                tokenizer_config=tokenizer_config,
                inference_config=inference_config,
                prompt="",
            )

    def test_whitespace_only_prompt_raises_value_error(
        self,
        mock_model: MagicMock,
        tokenizer_config: TokenizerConfig,
        inference_config: InferenceConfig,
    ) -> None:
        with pytest.raises(ValueError, match="prompt"):
            complete_prompt(
                model=mock_model,
                tokenizer_config=tokenizer_config,
                inference_config=inference_config,
                prompt="   ",
            )
