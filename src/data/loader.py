"""Utilities for loading and preprocessing text data for training.

Provides functions to load delimited text from files, calculate batch
counts, and build a Grain DataLoader from raw text blocks.
"""

import logging
from pathlib import Path

import grain.python as pygrain

logger = logging.getLogger(__name__)

from src.config import TokenizerConfig, TrainingConfig, ModelConfig
from src.paths import validate_project_path, format_path_for_display
from src.data.dataset import StoryDataset


def load_text_from_file(
    *,
    file_path: str | Path,
    delimiter: str,
    max_paragraphs: int | None = None
) -> list[str]:
    """
    Efficiently loads paragraphs from a text file.
    Paragraphs are loaded line by line to avoid loading the entire file into memory.

    Args:
        file_path: Path to the text file containing paragraphs
        delimiter: String that marks the end of a paragraph
        max_paragraphs: Maximum number of paragraphs to load (None for all)

    Returns:
        List of stripped text strings, one per delimited paragraph

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path attempts to escape the project root
    """

    # Validate path
    file_path = validate_project_path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Display path relative to project root for readability and privacy
    display_path = format_path_for_display(file_path)
    max_str = f"{max_paragraphs:,}" if max_paragraphs is not None else "all"
    logger.info(f"Loading data from {display_path} (max {max_str} paragraphs)")

    paragraphs, current_paragraph = [], []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if delimiter in line:
                parts = line.split(delimiter)
                for part in parts[:-1]:
                    current_paragraph.append(part)
                    story_text = ''.join(current_paragraph).strip()
                    if story_text:
                        paragraphs .append(story_text + delimiter)
                        if max_paragraphs and len(paragraphs) >= max_paragraphs:
                            break
                    current_paragraph = []

                if parts[-1].strip():
                    current_paragraph = [parts[-1]]
                else:
                    current_paragraph = []

                if max_paragraphs and len(paragraphs ) >= max_paragraphs:
                    break
            else:
                current_paragraph.append(line)

        if current_paragraph and (not max_paragraphs or len(paragraphs ) < max_paragraphs):
            story_text = ''.join(current_paragraph).strip()
            if story_text:
                paragraphs .append(story_text + delimiter)

    logger.info(f"Loaded {len(paragraphs)} paragraphs")
    return paragraphs


def calculate_batches(record_count: int, batch_size: int) -> int:
    """Compute and validate batches per epoch from dataset size and batch size.

    Args:
        record_count: Total number of records in the dataset
        batch_size: Number of records per batch

    Returns:
        Number of complete batches per epoch (integer division, remainder dropped)

    Raises:
        ValueError: if batches_per_epoch <= 0
    """
    batches_per_epoch = record_count // batch_size
    if batches_per_epoch <= 0:
        raise ValueError(
            f"Calculated {batches_per_epoch} batches per epoch but must be > 0. Training aborted."
        )
    logger.info(f"Calculated batches per epoch: {batches_per_epoch}")
    return batches_per_epoch

def preprocess_data(
    *,
    text_blocks: list[str],
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    training_config: TrainingConfig,
) -> pygrain.DataLoader:
    """Tokenize, pad, and batch text blocks into a Grain DataLoader.

    Args:
        text_blocks: List of delimiter-separated strings (one per paragraph)
        model_config: Model configuration; maxlen determines the padded sequence length
        tokenizer_config: Tokenizer settings (encoder, delimiter, pad token)
        training_config: Training hyperparameters (batch size, shuffle, seed)

    Returns:
        Grain DataLoader yielding batches of shape (batch_size, maxlen)

    Raises:
        ValueError: If no valid text blocks are found in the dataset
    """

    # Create dataset, which contains the raw data and knows how to tokenize
    # and pad each item into a fixed-length array of integers on demand.
    dataset = StoryDataset(
        stories=text_blocks,
        maxlen=model_config.maxlen,
        tokenizer_config=tokenizer_config
    )

    # Create sampler with sharding support. This decides the order
    # in which the dataset's indices are visited (shuffled or sequential).
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=training_config.shuffle,
        seed=training_config.seed,
        shard_options=pygrain.NoSharding(),

        # num_epochs parameter here is required by the pygrain API and it is analogous to repetitions per epoch.
        # This variable is different than the 'epochs' of the outer training loop, which are set in Trainer class.
        num_epochs=1,
    )

    # Create dataloader, which groups individual samples into batches for the training loop to consume later.
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=training_config.batch_size, drop_remainder=True)]
    )
    logger.info(f"Created DataLoader with batch_size={training_config.batch_size}, maxlen={model_config.maxlen}")

    return dataloader
