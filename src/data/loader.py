"""Data loading utilities for story datasets."""

import logging
from pathlib import Path

import grain.python as pygrain

logger = logging.getLogger(__name__)

from src.config import TokenizerConfig
from src.paths import validate_project_path, format_path_for_display
from src.data.dataset import StoryDataset


def load_text_from_file(
    file_path: str | Path,
    delimiter: str,
    max_paragraphs: int | None = None
) -> list[str]:
    """
    Efficiently loads paragraphs from a text file.
    Paragraphs are loaded line by line to avoid loading the entire file into memory.

    Args:
        file_path: Path to the text file containing stories
        delimiter: String that marks the end of a paragraph
        max_paragraphs: Maximum number of stories to load (None for all)

    Returns:
        List of text strings where each element represents a paragraph.
        In the case of TinyStories-1000.txt, each element is a story.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path attempts to escape the project root
    """

    # Validate path is within project root
    file_path = validate_project_path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Display path relative to project root for readability and privacy
    display_path = format_path_for_display(file_path)

    max_str = f"{max_paragraphs:,}" if max_paragraphs is not None else "all"
    logger.info("Loading data from %s (max %s paragraphs)", display_path, max_str)

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

    logger.info("Loaded %s paragraphs", f"{len(paragraphs):,}")
    return paragraphs


def preprocess_data(
    list_of_paragraphs: list[str],
    batch_size: int,
    maxlen: int,
    tokenizer_config: TokenizerConfig,
    shuffle: bool,
    seed: int,
) -> tuple[pygrain.DataLoader, int]:
    """
    Preprocess data with memory-efficient chunk reading.

    Args:
        list_of_paragraphs: list of delimiter-separated strings
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        tokenizer_config: Tokenizer configuration (encoder, delimiter, pad token)
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)

    Raises:
        ValueError: If no valid stories found in the dataset
    """

    total_size = len(list_of_paragraphs)
    if total_size == 0:
        raise ValueError("No valid stories found in the dataset")

    # Calculate estimated batches per epoch
    estimated_batches_per_epoch = total_size // batch_size
    logger.debug("Estimated batches per epoch: %s", f"{estimated_batches_per_epoch:,}")

    # Create efficient dataset
    dataset = StoryDataset(list_of_paragraphs, maxlen, tokenizer_config)

    # Configure sampler with sharding support
    #   num_epochs=1: one dataset pass per iteration; epoch repetition is the Trainer's responsibility
    #   num_epochs is the variable name set by pygrain library, but its more like 'repititions per epoch'
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=1, # dataset repetitions per DataLoader pass (pygrain API)
    )

    # Create DataLoader with efficient batching
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            pygrain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    logger.debug("Created DataLoader with batch_size=%d, maxlen=%d", batch_size, maxlen)
    return dataloader, estimated_batches_per_epoch
