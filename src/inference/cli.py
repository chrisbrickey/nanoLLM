"""
nanoLLM/src/inference/cli.py

CLI-specific inference utilities shared by scripts.
"""

import argparse
import dataclasses
from pathlib import Path

from src.config import InferenceConfig
from src.paths import CHECKPOINTS_DIR
from src.training.checkpoint import get_latest_checkpoint

_INFERENCE_OVERRIDE_FIELDS = ("max_new_tokens", "temperature", "seed")


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Register flags that inference scripts share in common.

    Optional flags default to None here so that we can distinguish
    between 'user did not pass this' and 'user specified an override'.
    """
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to complete")
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        default=None,
        help="Path to checkpoint bundle to load. If not specified, the most recent checkpoint is loaded.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=f"Maximum tokens to generate (default: {InferenceConfig.max_new_tokens})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature; must be > 0 (default: {InferenceConfig.temperature})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: non-deterministic)",
    )


def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    """Apply non-None CLI overrides on top of InferenceConfig defaults."""
    overrides: dict[str, object] = {
        field: getattr(args, field)
        for field in _INFERENCE_OVERRIDE_FIELDS
        if getattr(args, field) is not None
    }
    return dataclasses.replace(InferenceConfig(), **overrides)


def resolve_source_checkpoint(args: argparse.Namespace) -> Path:
    """Return the checkpoint path to load, defaulting to the most recent checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint source is provided and none exists.
    """
    if args.checkpoint_source:
        return Path(args.checkpoint_source)

    latest = get_latest_checkpoint(CHECKPOINTS_DIR)
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINTS_DIR}.")
    return latest
