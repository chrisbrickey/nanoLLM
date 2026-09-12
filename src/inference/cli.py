"""
nanoLLM/src/inference/cli.py

CLI-specific inference utilities shared by scripts.
"""

import argparse
import logging

from src.cli import apply_cli_overrides
from src.config import (
    MIN_NEW_TOKENS,
    MIN_TEMPERATURE_EXCLUSIVE,
    RECOMMENDED_NEW_TOKENS,
    RECOMMENDED_TEMPERATURE,
    InferenceConfig,
    ParamRange,
)

logger = logging.getLogger(__name__)


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
        help=(
            f"Maximum tokens to generate; minimum value permitted is {MIN_NEW_TOKENS}, "
            f"recommended range {RECOMMENDED_NEW_TOKENS.minimum} to {RECOMMENDED_NEW_TOKENS.maximum} "
            f"(default: {InferenceConfig.max_new_tokens})"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            f"Sampling temperature; must be > {MIN_TEMPERATURE_EXCLUSIVE}, recommended range "
            f"{RECOMMENDED_TEMPERATURE.minimum} to {RECOMMENDED_TEMPERATURE.maximum} "
            f"(default: {InferenceConfig.temperature})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling; Default is None, which results in non-deterministic behavior",
    )


def _warn_if_outside_recommended(name: str, value: float, recommended: ParamRange) -> None:
    """Log a warning for an out-of-band value. Recommended bands never block or clamp."""
    if not recommended.contains(value):
        logger.warning(
            f"{name} {value} is outside the recommended range of "
            f"{recommended.minimum} to {recommended.maximum}. "
            "Output quality may degrade."
        )


def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    """Apply non-None CLI overrides on top of InferenceConfig defaults."""
    config = apply_cli_overrides(InferenceConfig(), args)

    _warn_if_outside_recommended("Temperature", config.temperature, RECOMMENDED_TEMPERATURE)
    _warn_if_outside_recommended("Max new tokens", config.max_new_tokens, RECOMMENDED_NEW_TOKENS)

    return config
