"""Tokenize, pad, and batch raw text blocks into a Grain DataLoader."""

import logging

import grain.python as pygrain

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.dataset import StoryDataset

logger = logging.getLogger(__name__)


class Processor:
    """Builds a Grain DataLoader from raw text blocks.

    This class operates on three configs that are stable
    for the lifetime of a training run and are bound at
    construction time

    .process() is invoked per dataset of text blocks
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        tokenizer_config: TokenizerConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize the processor with the configs used for every batch.

        Args:
            model_config: Model configuration; maxlen determines padded sequence length
            tokenizer_config: Tokenizer settings (encoder, delimiter, pad token)
            training_config: Training hyperparameters (batch size, shuffle, seed)
        """
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.training_config = training_config

    def process(self, text_blocks: list[str]) -> pygrain.DataLoader:
        """Tokenize, pad, and batch text blocks into a Grain DataLoader.

        Args:
            text_blocks: List of delimiter-separated strings (one per paragraph)

        Returns:
            Grain DataLoader yielding batches of shape (batch_size, maxlen)
        """

        # Create dataset, which contains the raw data and knows how to tokenize
        # and pad each item into a fixed-length array of integers on demand.
        dataset = StoryDataset(
            stories=text_blocks,
            maxlen=self.model_config.maxlen,
            tokenizer_config=self.tokenizer_config,
        )

        # Create sampler with sharding support. This decides the order
        # in which the dataset's indices are visited (shuffled or sequential).
        sampler = pygrain.IndexSampler(
            num_records=len(dataset),
            shuffle=self.training_config.shuffle,
            seed=self.training_config.seed,
            shard_options=pygrain.NoSharding(),

            # num_epochs parameter here is required by the pygrain API
            # It is analogous to repetitions per epoch.
            # This variable is different than the 'epochs' of the outer training loop,
            # which are set in Trainer class.
            num_epochs=1,
        )

        # Create dataloader, which groups individual samples into batches
        # for the training loop to consume later.
        dataloader = pygrain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            operations=[
                pygrain.Batch(
                    batch_size=self.training_config.batch_size,
                    drop_remainder=True,
                )
            ],
        )
        logger.info(
            f"Created DataLoader with batch_size={self.training_config.batch_size}, "
            f"maxlen={self.model_config.maxlen}"
        )

        return dataloader
