"""
nanoLLM/scripts/train.py

CLI entry point for training an untrained nanoLLM.

This script loads an untrained model from configs
and runs N epochs of training.

This script is not capable of loading pre-trained weights.
To continue training in subsequent phases, use the `nanollm-resume`
script, which loads pre-trained weights from a checkpoint
and then applies the same training functionality.

Usage:
    # fresh training with default configuration
    uv run nanollm-train

    # fresh training with example overrides
    uv run nanollm-train --epochs 5 --batch-size 64 --checkpoint-destination my_run.orbax
"""

import argparse
import logging
import sys

from src.config import ModelConfig, TokenizerConfig
from src.logging_setup import setup_logging
from src.model.model import NanoLLM, count_params
from src.training.cli import (
    add_shared_training_args,
    build_training_config,
    resolve_data_file,
    resolve_destination_checkpoint,
)
from src.training.runner import Runner

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an untrained nanoLLM.")
    add_shared_training_args(parser)
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
        # Load fresh, untrained model from configs
        logger.info("Building untrained model...")
        model_config = ModelConfig()
        model = NanoLLM(model_config)
        tokenizer_config = TokenizerConfig()
        logger.info(f"Model ready ({count_params(model)} parameters)")
    except Exception as e:
        logger.error(f"Failed to build untrained model from configs: {e}")
        sys.exit(1)


    # --- Execute Operations ---


    try:
        runner = Runner(
            model=model,
            tokenizer_config=tokenizer_config,
            data_source=data_source,
            training_config=training_config,
            checkpoint_destination=checkpoint_destination,
        )
        runner.run()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
