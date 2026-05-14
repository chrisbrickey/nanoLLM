"""
nanoLLM/src/training/runner.py

Orchestrates data loading, processing, and training run.
This module can be used by all types of entry points
(e.g., CLI scripts, notebooks) because it consumes
agnostic, typed configurations.
"""

import logging
from pathlib import Path

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.loader import load_text_from_file, preprocess_data, calculate_batches
from src.model.model import NanoLLM
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def run(
    *,
    model: NanoLLM,
    tokenizer_config: TokenizerConfig,
    data_source: Path,
    training_config: TrainingConfig,
    checkpoint_destination: Path,

    # optional parameters used only if loading pre-trained weights from checkpoint
    checkpoint_source: Path | None = None,
    previous_epochs_completed: int = 0,

) -> None:
    """Shared boilerplate for training pathways: loads data, preprocesses it, and trains.

    Args:
        model: The NanoLLM model to train
        tokenizer_config: Tokenizer settings used for data preprocessing
        data_source: Path to the raw text file
        training_config: Training hyperparameters
        checkpoint_destination: Where to write the checkpoint after training
        checkpoint_source: Source checkpoint to resume from; must be paired with
            previous_epochs_completed or the cumulative epoch count will be wrong
        previous_epochs_completed: Epochs already completed in prior runs; only
            meaningful when checkpoint_source is provided

    Raises:
        FileNotFoundError: If data_source does not exist
        ValueError: If the dataset is empty or yields no complete batches
    """

    # Load the data
    logger.info("Loading data ...")
    stories = load_text_from_file(
        file_path=data_source,
        delimiter=tokenizer_config.delimiter,
        max_paragraphs=training_config.max_stories,
    )
    logger.info("Data loading complete.")

    # Validate data characteristics; Cross-domain considerations (not strictly loading or training)
    record_count = len(stories)
    if record_count == 0:
        raise ValueError("Dataset is empty. Training aborted.")
    batches_per_epoch = calculate_batches(record_count, training_config.batch_size)

    # Preprocess the data
    logger.info("Processing data ...")
    dataloader = preprocess_data(
        text_blocks=stories,
        model_config=model.config,
        tokenizer_config=tokenizer_config,
        training_config=training_config,
    )
    logger.info("Data processing complete.")

    # Train the model
    logger.info(_format_banner("Commencing training..."))
    trainer = Trainer(
        model=model,
        data_source=data_source,
        dataloader=dataloader,
        batches_per_epoch=batches_per_epoch,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        checkpoint_destination=checkpoint_destination,
        checkpoint_source=checkpoint_source,
        previous_epochs_completed=previous_epochs_completed,
    )
    trainer.train()
    logger.info(_format_banner("Training complete."))

def _format_banner(text: str, width: int = 30) -> str:
    """Wrap a short message in dashed top/bottom banner lines with breathing room."""
    edge = "-" * width
    return f"\n\n{edge}\n{text}\n{edge}\n\n"