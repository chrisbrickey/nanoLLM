"""Integration tests for the data pipeline: StoryDataset, Processor,
real tiktoken tokenizer, and pygrain sampler/batcher."""

import numpy as np
import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.dataset import StoryDataset
from src.data.processor import Processor

SAMPLE_STORY_001 = "sample story 001"
SAMPLE_STORY_002 = "sample story 002 with additional content here"
SAMPLE_STORY_003 = "sample story 003"
TEST_DELIMITER = "<|endoftext|>"
TEST_MAXLEN = 20
TOKENIZER_NAME = "gpt2"
PAD_TOKEN_ID = 0

# Processor-pipeline knobs
PIPELINE_BATCH_SIZE = 4
PIPELINE_NUM_STORIES = 12  # exactly 3 full batches at batch_size=4
PIPELINE_SEED = 42


@pytest.fixture
def sample_stories() -> list[str]:
    return [SAMPLE_STORY_001, SAMPLE_STORY_002, SAMPLE_STORY_003]


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        delimiter=TEST_DELIMITER, name=TOKENIZER_NAME, pad_token_id=PAD_TOKEN_ID
    )


@pytest.fixture
def dataset(
    sample_stories: list[str], tokenizer_config: TokenizerConfig
) -> StoryDataset:
    return StoryDataset(
        stories=sample_stories,
        maxlen=TEST_MAXLEN,
        tokenizer_config=tokenizer_config,
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        maxlen=TEST_MAXLEN,
        vocab_size=50,
        embed_dim=12,
        num_heads=3,
        feed_forward_dim=16,
        num_transformer_blocks=1,
    )


@pytest.fixture
def pipeline_stories() -> list[str]:
    return [f"sample story number {i:03d}" for i in range(PIPELINE_NUM_STORIES)]


def _materialize_batches(processor: Processor, stories: list[str]) -> list[np.ndarray]:
    """Run processor.process(stories) and convert each yielded batch to a numpy array."""
    return [np.asarray(batch) for batch in processor.process(stories)]


class TestStoryDatasetIntegration:
    """Integration tests for StoryDataset with real tiktoken tokenizer."""

    def test_len_returns_story_count(self, dataset: StoryDataset) -> None:
        assert len(dataset) == 3

    def test_getitem_returns_fixed_length_token_list(self, dataset: StoryDataset) -> None:
        result = dataset[0]
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)
        assert len(result) == TEST_MAXLEN

    def test_all_items_have_correct_length(self, dataset: StoryDataset) -> None:
        for i in range(len(dataset)):
            assert len(dataset[i]) == TEST_MAXLEN

    def test_truncates_long_story(self, tokenizer_config: TokenizerConfig) -> None:
        long_story = "word " * 100
        ds = StoryDataset(
            stories=[long_story],
            maxlen=TEST_MAXLEN,
            tokenizer_config=tokenizer_config,
        )
        assert len(ds[0]) == TEST_MAXLEN

    def test_pads_short_story_with_pad_token(
        self, tokenizer_config: TokenizerConfig
    ) -> None:
        short_story = "hi"
        ds = StoryDataset(
            stories=[short_story],
            maxlen=TEST_MAXLEN,
            tokenizer_config=tokenizer_config,
        )
        result = ds[0]
        assert len(result) == TEST_MAXLEN
        assert result[-1] == PAD_TOKEN_ID


class TestProcessorIntegration:
    """End-to-end integration tests for Processor: real tiktoken tokenizer,
    real StoryDataset, real pygrain sampler/batcher."""

    def test_yields_batches_with_expected_shape(
        self,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
        pipeline_stories: list[str],
    ) -> None:
        training_config = TrainingConfig(
            batch_size=PIPELINE_BATCH_SIZE, shuffle=False, seed=PIPELINE_SEED
        )
        processor = Processor(
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            training_config=training_config,
        )

        batches = _materialize_batches(processor, pipeline_stories)

        expected_batch_count = PIPELINE_NUM_STORIES // PIPELINE_BATCH_SIZE
        assert len(batches) == expected_batch_count
        for batch in batches:
            # pygrain stacks each item as a column, so the raw batch shape is
            # (maxlen, batch_size). Trainer transposes downstream.
            assert batch.shape == (TEST_MAXLEN, PIPELINE_BATCH_SIZE)
            assert np.issubdtype(batch.dtype, np.integer)

    def test_drops_remainder_batch(
        self,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
    ) -> None:
        """With drop_remainder=True (Processor's default behavior), trailing
        stories that don't fill a complete batch must not be yielded."""
        partial_count = PIPELINE_BATCH_SIZE * 2 + 1  # 9 stories at batch_size=4 -> 2 full
        stories = [f"sample story number {i:03d}" for i in range(partial_count)]
        training_config = TrainingConfig(
            batch_size=PIPELINE_BATCH_SIZE, shuffle=False, seed=PIPELINE_SEED
        )
        processor = Processor(
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            training_config=training_config,
        )

        batches = _materialize_batches(processor, stories)
        assert len(batches) == partial_count // PIPELINE_BATCH_SIZE

    def test_same_seed_produces_same_batch_sequence(
        self,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
        pipeline_stories: list[str],
    ) -> None:
        """Two Processor runs with shuffle=True and identical seeds must
        produce byte-identical batches."""
        training_config = TrainingConfig(
            batch_size=PIPELINE_BATCH_SIZE, shuffle=True, seed=PIPELINE_SEED
        )

        first_run = _materialize_batches(
            Processor(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                training_config=training_config,
            ),
            pipeline_stories,
        )
        second_run = _materialize_batches(
            Processor(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                training_config=training_config,
            ),
            pipeline_stories,
        )

        assert len(first_run) == len(second_run)
        for first_batch, second_batch in zip(first_run, second_run):
            np.testing.assert_array_equal(first_batch, second_batch)

    def test_shuffle_changes_order_versus_unshuffled(
        self,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
        pipeline_stories: list[str],
    ) -> None:
        """With shuffle=False the rows reflect the input order; with shuffle=True
        the seed used here yields a different row order. Asserting concrete
        difference rules out a silently-ignored shuffle flag."""
        unshuffled = _materialize_batches(
            Processor(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                training_config=TrainingConfig(
                    batch_size=PIPELINE_BATCH_SIZE, shuffle=False, seed=PIPELINE_SEED
                ),
            ),
            pipeline_stories,
        )
        shuffled = _materialize_batches(
            Processor(
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                training_config=TrainingConfig(
                    batch_size=PIPELINE_BATCH_SIZE, shuffle=True, seed=PIPELINE_SEED
                ),
            ),
            pipeline_stories,
        )

        unshuffled_rows = np.concatenate(unshuffled, axis=0)
        shuffled_rows = np.concatenate(shuffled, axis=0)
        assert not np.array_equal(unshuffled_rows, shuffled_rows)
