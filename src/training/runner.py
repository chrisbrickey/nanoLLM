"""
nanoLLM/src/training/runner.py

Orchestrates data loading, processing, and training run.
This module can be used by all types of entry points
(e.g., CLI scripts, notebooks) because it consumes
agnostic, typed configurations.
"""

import dataclasses
import logging
from pathlib import Path

from src.checkpoint import CheckpointMetadata, save_checkpoint
from src.config import TokenizerConfig, TrainingConfig
from src.data.loader import load_text_from_file, preprocess_data, calculate_batches
from src.model.model import NanoLLM
from src.training.resume_context import ResumeContext
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def run(
    *,
    model: NanoLLM,
    tokenizer_config: TokenizerConfig,
    data_source: Path,
    training_config: TrainingConfig,
    checkpoint_destination: Path | None,
    resume_from: ResumeContext | None = None,
) -> None:
    """Shared boilerplate for training pathways:
    loads data, preprocesses it, trains the model.

    It also manages loading weights from a checkpoint
    and persisting a new checkpoint after training run.

    Args:
        For execution of training...
        - model: the model to train
        - tokenizer_config: tokenizer settings used for data preprocessing
        - data_source: path to the raw text file
        - training_config: training parameters

        For checkpoint loading and persistence...
        - checkpoint_destination: where to write the checkpoint after training;
                            if None, checkpoint is not persisted
        - resume_from: ResumeContext with the data required when resuming a prior run;
                            if None, checkpoint (pre-trained) weights is not loaded

    Raises:
        FileNotFoundError: if data_source does not exist
        ValueError: if the dataset is empty or yields no complete batches
    """

    # Log checkpoint characteristics
    if checkpoint_destination is None:
        logger.warning("No checkpoint_destination path provided so no checkpoint will be persisted.")
    if resume_from is None:
        logger.info("No resume context received so no pre-trained weights will be loaded.")

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

    # Log training configuration
    previous_epochs_completed, checkpoint_source = _unpack_resume_context(resume_from)
    header_lines = _build_training_header_lines(
        data_source=data_source,
        training_config=training_config,
        checkpoint_destination=checkpoint_destination,
        checkpoint_source=checkpoint_source,
        previous_epochs=previous_epochs_completed,
    )
    logger.info("\n\n%s\n\n", "\n".join(header_lines))

    # Train the model
    logger.info(_format_banner("Commencing training..."))
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        batches_per_epoch=batches_per_epoch,
        training_config=training_config,
    )
    metrics_history = trainer.train()
    logger.info(_format_banner("Training complete."))

    # Persist checkpoint
    cumulative_epochs_completed = previous_epochs_completed + training_config.epochs
    logger.info(
        f"\n\tAll {training_config.epochs} epochs completed.\n"
        f"\tThis is in addition to {previous_epochs_completed} epochs accumulated during previous trainings.\n"
        f"\tCumulative epochs completed: {cumulative_epochs_completed }.\n"
    )
    _persist_checkpoint(
        model=model,
        destination=checkpoint_destination,
        metrics_history=metrics_history,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        cumulative_epochs_completed=cumulative_epochs_completed,
    )


def _format_banner(text: str, width: int = 30) -> str:
    """Wrap a short message in dashed top/bottom banner lines with breathing room."""
    edge = "-" * width
    return f"\n\n{edge}\n{text}\n{edge}\n\n"


def _build_training_header_lines(
    *,
    data_source: Path | None,
    training_config: TrainingConfig,
    checkpoint_destination: Path | None,
    checkpoint_source: Path | None = None,
    previous_epochs: int = 0,
) -> list[str]:
    """Build a per-invocation summary header as a list of lines.

    Args:
        data_source: Path to the training data file
        training_config: Training hyperparameters for this run
        checkpoint_destination: Where the checkpoint will be written
        checkpoint_source: Source checkpoint path; if provided, resume-related
            lines are prepended to the header
        previous_epochs: Number of epochs completed before this run
    """

    lines = []
    lines.extend(
        [
            f"\tepochs (this run):      {training_config.epochs}",
            f"\tdata source:            {data_source}",
            f"\tmax stories:            {training_config.max_stories}",
            f"\tbatch size:             {training_config.batch_size}",
            f"\tshuffle:                {training_config.shuffle}",
            f"\tseed:                   {training_config.seed}",
            "",
            f"\tcheckpoint source:       {checkpoint_source}",
            f"\tprevious epochs trained: {previous_epochs}",
            "",
            f"\tcheckpoint destination: {checkpoint_destination}",
        ]
    )
    return lines


def _unpack_resume_context(
    resume_from: ResumeContext | None,
) -> tuple[int, Path | None]:
    """Return (previous_epochs_completed, checkpoint_source), defaulting to (0, None)
    when no resume context is provided."""
    if resume_from is None:
        return 0, None
    return resume_from.previous_epochs_completed, resume_from.source

def _persist_checkpoint(
    *,
    model: NanoLLM,
    destination: Path | None,
    metrics_history: dict[str, list[float]],
    training_config: TrainingConfig,
    tokenizer_config: TokenizerConfig,
    cumulative_epochs_completed: int,
) -> None:
    """Assemble checkpoint metadata and persist the model.

    Skips persistence (logging only) when destination is None.
    """
    if destination is None:
        logger.info("Checkpoint path undefined. No checkpoint persisted.")
        return

    train_losses = metrics_history.get("train_loss", [])
    final_loss = train_losses[-1] if train_losses else None
    metadata = CheckpointMetadata(
        cumulative_epochs_completed=cumulative_epochs_completed,
        final_loss=final_loss,
        model_config=dataclasses.asdict(model.config),
        training_config=dataclasses.asdict(training_config),
        tokenizer_config=dataclasses.asdict(tokenizer_config),
    )
    save_checkpoint(model, destination, metadata=metadata)
