"""
nanoLLM/scripts/resume.py

CLI entry point for resuming training of nanoLLM from a checkpoint.

Loads an existing checkpoint bundle (including weights, configs, metadata)
and runs N more epochs of training. Persists a new checkpoint bundle whose
cumulative_epochs_completed reflects the running total of epochs trained.

To train a model from an untrained state (without loading a checkpoint),
use the `nanollm-train` script instead.

Usage:
    # resume training with default configuration (discovers and loads most recent checkpoint)
    uv run nanollm-resume

    # resume training with example overrides
    uv run nanollm-resume --epochs 5
    uv run nanollm-resume --checkpoint-source path/to/bundle --checkpoint-destination path/to/new
"""

import argparse
import logging
import sys
from pathlib import Path

from src.checkpoint import (
    build_model_from_checkpoint,
    get_latest_checkpoint,
)
from src.logging_setup import setup_logging
from src.model.model import count_params
from src.paths import CHECKPOINTS_DIR
from src.training.cli import (
    add_shared_training_args,
    build_training_config,
    resolve_data_file,
    resolve_destination_checkpoint,
)
from src.training.resume_context import ResumeContext
from src.training.runner import run

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume nanoLLM training from a checkpoint.")
    add_shared_training_args(parser)

    # Add arguments that are unique to this script
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        default=None,
        help=("Path to checkpoint bundle from which to load pre-trained weights. If not specified, the most recent checkpoint is loaded."),
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()


    # --- Prepare Inputs ---


    try:
        # Construct configs from CLI arguments
        training_config = build_training_config(args)
        data_source = resolve_data_file(args)
        checkpoint_destination = resolve_destination_checkpoint(args)
    except Exception as e:
        logger.error(f"Failed to construct configs from CLI arguments: {e}")
        sys.exit(1)

    try:
        # Resolve checkpoint source bundle (defaults to most recent)
        if args.checkpoint_source:
            source_path: Path | None = Path(args.checkpoint_source)
        else:
            source_path = get_latest_checkpoint(CHECKPOINTS_DIR)
            if source_path is None:
                raise FileNotFoundError(f"No checkpoints found in {CHECKPOINTS_DIR}.")
    except Exception as e:
        logger.error(f"Failed to resolve checkpoint source bundle: {e}")
        sys.exit(1)

    try:
        # Retrieve and apply data from checkpoint bundle
        # These methods look at both the standard orbax file structure and the non-standard json sidecar (METADATA.json).

        # Load trained model
        logger.info(f"Loading checkpoint from {source_path}")
        model, model_config, tokenizer_config = build_model_from_checkpoint(source_path)
        logger.info(f"Model ready ({count_params(model)} parameters)")

        # Pair the source with its cumulative epoch count in one step to avoid divergence
        resume_ctx = ResumeContext.from_checkpoint(source_path)
    except Exception as e:
        logger.error(f"Failed to retrieve and apply data from checkpoint bundle: {e}")
        sys.exit(1)


    # --- Execute Operations ---


    try:
        run(
            model=model,
            tokenizer_config=tokenizer_config,
            data_source=data_source,
            training_config=training_config,
            checkpoint_destination=checkpoint_destination,
            resume_from=resume_ctx,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
