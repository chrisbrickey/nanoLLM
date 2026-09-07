"""
nanoLLM/src/inference/cli.py

CLI-specific inference utilities shared by scripts.
"""

import argparse

from src.cli import apply_cli_overrides
from src.config import InferenceConfig


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
    return apply_cli_overrides(InferenceConfig(), args)
