"""
nanoLLM/scripts/train.py

CLI entry point for training nanoLLM.

Usage:
    uv run nanollm-train
    uv run nanollm-train --num-epochs 5 --batch-size 64 --checkpoint my_run.orbax
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import flax.nnx as nnx

from src.config import (
    DEFAULT_CHECKPOINT_PATH,
    TINYSTORIES_FILE,
    TokenizerConfig,
    model_config,
    training_config,
)
from src.data.loader import load_text_from_file, preprocess_data
from src.model.model import NanoLLM
from src.training.trainer import Trainer

_DELIMITER = "<|endoftext|>"
_TOKENIZER_NAME = "gpt2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nanoLLM from the command line.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
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
    args = _parse_args()

    overrides: dict[str, object] = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.max_stories is not None:
        overrides["max_stories"] = args.max_stories
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.shuffle is not None:
        overrides["shuffle"] = args.shuffle

    config = dataclasses.replace(training_config, **overrides)

    data_file = Path(args.data_file) if args.data_file else TINYSTORIES_FILE
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_PATH

    print("=" * 50)
    print("nanoLLM training")
    print(f"  data file:   {data_file}")
    print(f"  max stories: {config.max_stories}")
    print(f"  epochs:      {config.num_epochs}")
    print(f"  batch size:  {config.batch_size}")
    print(f"  shuffle:     {config.shuffle}")
    print(f"  seed:        {config.seed}")
    print(f"  checkpoint:  {checkpoint_path}")
    print("=" * 50)

    tokenizer_config = TokenizerConfig(delimiter=_DELIMITER, name=_TOKENIZER_NAME)

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
            delimiter=tokenizer_config.delimiter,
            num_epochs=config.num_epochs,
            shuffle=config.shuffle,
            seed=config.seed,
        )

        print("\nBuilding model ...")
        model = NanoLLM(
            maxlen=model_config.maxlen,
            vocab_size=model_config.vocab_size,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            feed_forward_dim=model_config.feed_forward_dim,
            num_transformer_blocks=model_config.num_transformer_blocks,
            rngs=nnx.Rngs(config.seed),
        )
        print("Model ready.\n")

        trainer = Trainer(
            model=model,
            training_config=config,
            dataloader=dataloader,
            batches_per_epoch=batches_per_epoch,
            checkpoint_path=checkpoint_path,
        )
        trainer.train()

        print("\nTraining complete.")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
