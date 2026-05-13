"""
nanoLLM/src/training/runner.py

Helper methods that orchestrate portions of the training
process given real config objects. These helpers can be
used for all types of entry points (e.g., CLI scripts,
notebooks) because they consume typed configurations.
"""

import logging
import sys
from pathlib import Path

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.loader import load_text_from_file, preprocess_data
from src.model.model import NanoLLM
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def prepare_dataloader(
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    data_file: Path,
    training_config: TrainingConfig,
):
    """Load text → preprocess → DataLoader. Returns (dataloader, batches_per_epoch)."""
    logger.info("Loading data ...")
    stories = load_text_from_file(
        data_file,
        delimiter=tokenizer_config.delimiter,
        max_paragraphs=training_config.max_stories,
    )
    logger.info("Data loading complete.")

    logger.info("Processing data ...")
    dataloader, batches_per_epoch = preprocess_data(
        stories,
        batch_size=training_config.batch_size,
        maxlen=model_config.maxlen,
        tokenizer_config=tokenizer_config,
        shuffle=training_config.shuffle,
        seed=training_config.seed,
    )
    logger.info("Data processing complete.")
    return dataloader, batches_per_epoch


def execute_training_run(
    *,
    model: NanoLLM,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    data_source: Path,
    training_config: TrainingConfig,
    checkpoint_destination: Path,

    # optional parameters if loaded model from checkpoint
    checkpoint_source: Path | None = None,
    previous_epochs_completed: int = 0,

) -> None:
    """Shared boilerplate for training pathways:
    prepares the dataloader and runs training."""

    # Load and preprocess data
    dataloader, batches_per_epoch = prepare_dataloader(
        model_config, tokenizer_config, data_source, training_config
    )

    # Train the model
    trainer = Trainer(
        model=model,
        data_source=data_source,
        dataloader=dataloader,
        batches_per_epoch=batches_per_epoch,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        checkpoint_path=checkpoint_destination,
        checkpoint_source=checkpoint_source,
        previous_epochs_completed=previous_epochs_completed,
    )
    trainer.train()