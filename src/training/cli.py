"""
nanoLLM/src/training/cli.py

Argparse plumbing shared by scripts.

This module is CLI-specific because every function takes
(or produces) an argparse.Namespace.
"""

import argparse
import dataclasses
from pathlib import Path

from src.checkpoint import default_checkpoint_path
from src.config import TrainingConfig
from src.model.model import NanoLLM
from src.paths import DEFAULT_DATA_FILE


_TRAINING_OVERRIDE_FIELDS = ("batch_size", "epochs", "max_stories", "seed", "shuffle")


def add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    """Register flags that training scripts share in common.

    All flags default to None here (instead of system defaults that will
    eventually be assigned in fallback logic) so that we can distinguish
    between 'user did not pass this' and 'user specified an override'.
    """
    parser.add_argument("--data-file", type=str, default=None, help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=None)
    parser.add_argument(
        "--checkpoint-destination",
        type=str,
        default=None,
        help="Path to save checkpoint bundle directory (default: checkpoints/NanoLLM_{timestamp}/)",
    )


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Apply non-None CLI overrides on top of TrainingConfig defaults."""
    overrides: dict[str, object] = {
        field: getattr(args, field)
        for field in _TRAINING_OVERRIDE_FIELDS
        if getattr(args, field) is not None
    }
    return dataclasses.replace(TrainingConfig(), **overrides)


def resolve_data_file(args: argparse.Namespace) -> Path:
    return Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE


def resolve_destination_checkpoint(args: argparse.Namespace) -> Path:
    return (
        Path(args.checkpoint_destination)
        if args.checkpoint_destination
        else default_checkpoint_path(NanoLLM.__name__)
    )
