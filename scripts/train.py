"""
nanoLLM/scripts/train.py

CLI entry point for training nanoLLM.

Usage:
    uv run nanollm-train
    uv run nanollm-train --epochs 5 --batch-size 64 --checkpoint my_run.orbax
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import flax.nnx as nnx
import jax

from src.config import (
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)
from src.checkpoint import default_checkpoint_path
from src.paths import DEFAULT_DATA_FILE
from src.logging_setup import setup_logging
from src.data.loader import load_text_from_file, preprocess_data
from src.model.model import NanoLLM
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nanoLLM from the command line.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to save checkpoint (default: checkpoints/nano_checkpoint.orbax)",
    )
    parser.add_argument("--data-file", type=str, default=None, help="Path to training data file")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()

    overrides: dict[str, object] = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.max_stories is not None:
        overrides["max_stories"] = args.max_stories
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.shuffle is not None:
        overrides["shuffle"] = args.shuffle

    config = dataclasses.replace(TrainingConfig(), **overrides)

    data_file = Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(NanoLLM.__name__)

    tokenizer_config = TokenizerConfig()
    model_config = ModelConfig()

    logger.info(
        f"\n\n{'-' * 30}\n"
        "Commencing nanoLLM training:\n"
        "\tdata file:   %s\n"
        "\tmax stories: %s\n"
        "\tepochs:      %d\n"
        "\tbatch size:  %d\n"
        "\tshuffle:     %s\n"
        "\tseed:        %d\n"
        "\tcheckpoint:  %s\n"
        f"{'-' * 30}\n\n",
        data_file, config.max_stories, config.epochs,
        config.batch_size, config.shuffle, config.seed, checkpoint_path,
    )

    try:
        stories = load_text_from_file(
            data_file,
            delimiter=tokenizer_config.delimiter,
            max_paragraphs=config.max_stories,
        )

        dataloader, batches_per_epoch = preprocess_data(
            stories,
            batch_size=config.batch_size,
            maxlen=model_config.maxlen,
            tokenizer_config=tokenizer_config,
            shuffle=config.shuffle,
            seed=config.seed,
        )

        logger.info("Building model ...")
        model = NanoLLM(model_config)
        params = nnx.state(model, nnx.Param)
        param_count = sum(v.size for v in jax.tree_util.tree_leaves(params))
        logger.info("Model ready — %s parameters", f"{param_count:,}")

        trainer = Trainer(
            model=model,
            training_config=config,
            dataloader=dataloader,
            batches_per_epoch=batches_per_epoch,
            checkpoint_path=checkpoint_path,
        )
        trainer.train()

        logger.info(
            f"\n\n{'-' * 30}\n"
            "Training complete.\n"
            f"{'-' * 30}\n\n"
        )

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
