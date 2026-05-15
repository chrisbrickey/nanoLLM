"""Integration tests for Processor: StoryDataset + pygrain pipeline."""

import logging
from collections.abc import Generator
from unittest.mock import MagicMock, patch
import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.processor import Processor

TEST_DELIMITER = "<|endoftext|>"
TEST_BATCH_SIZE = 4
TEST_MAXLEN = 20
NUM_SAMPLE_PARAGRAPHS = 12

SAMPLE_MODEL_CONFIG = ModelConfig(
    maxlen=TEST_MAXLEN,
    vocab_size=50,
    embed_dim=12,
    num_heads=3,
    feed_forward_dim=16,
    num_transformer_blocks=1,
)
SAMPLE_TRAINING_CONFIG = TrainingConfig(batch_size=TEST_BATCH_SIZE, shuffle=False, seed=42)


@pytest.fixture
def sample_paragraphs() -> list[str]:
    return [f"paragraph_{i:03d}" for i in range(NUM_SAMPLE_PARAGRAPHS)]


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(delimiter=TEST_DELIMITER, name="gpt2")


class TestProcessor:
    """Integration tests for Processor: exercises StoryDataset + pygrain together."""

    @pytest.fixture(autouse=True)
    def _capture_logs(self, caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
        with caplog.at_level(logging.DEBUG, logger="src.data.processor"):
            yield

    def test_creates_dataloader_with_valid_data(
        self,
        sample_paragraphs: list[str],
        tokenizer_config: TokenizerConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch("src.data.processor.StoryDataset") as mock_dataset_class:
            with patch("src.data.processor.pygrain.IndexSampler"):
                with patch("src.data.processor.pygrain.DataLoader") as mock_loader_cls:
                    mock_dataset = MagicMock()
                    mock_dataset.__len__ = MagicMock(return_value=NUM_SAMPLE_PARAGRAPHS)
                    mock_dataset_class.return_value = mock_dataset

                    processor = Processor(
                        model_config=SAMPLE_MODEL_CONFIG,
                        tokenizer_config=tokenizer_config,
                        training_config=SAMPLE_TRAINING_CONFIG,
                    )
                    result = processor.process(sample_paragraphs)

        assert result is mock_loader_cls.return_value
        mock_dataset_class.assert_called_once_with(
            stories=sample_paragraphs,
            maxlen=TEST_MAXLEN,
            tokenizer_config=tokenizer_config,
        )
        assert "Created DataLoader" in caplog.text

    def test_shuffle_and_seed_passed_to_sampler(
        self, sample_paragraphs: list[str], tokenizer_config: TokenizerConfig
    ) -> None:
        shuffle_config = TrainingConfig(batch_size=TEST_BATCH_SIZE, shuffle=True, seed=99)
        with patch("src.data.processor.StoryDataset") as mock_dataset_class:
            with patch("src.data.processor.pygrain.IndexSampler") as mock_sampler:
                with patch("src.data.processor.pygrain.DataLoader"):
                    mock_dataset = MagicMock()
                    mock_dataset.__len__ = MagicMock(return_value=NUM_SAMPLE_PARAGRAPHS)
                    mock_dataset_class.return_value = mock_dataset

                    processor = Processor(
                        model_config=SAMPLE_MODEL_CONFIG,
                        tokenizer_config=tokenizer_config,
                        training_config=shuffle_config,
                    )
                    processor.process(sample_paragraphs)

        call_kwargs = mock_sampler.call_args.kwargs
        assert call_kwargs["shuffle"] is True
        assert call_kwargs["seed"] == 99
