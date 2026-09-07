"""
nanoLLM/scripts/generate.py

CLI entry point for generating a text completion from nanoLLM.

Loads an existing checkpoint bundle (including weights, configs, metadata)
and generates a completion for a given text prompt.

Usage:
  # generate text using defaults (most recent checkpoint)
  uv run nanollm-generate --prompt "The cat's eyes"

  # generate text using overrides
  uv run nanollm-generate --prompt "The cat's eyes" --checkpoint-source checkpoints/bundle_X/ --max-new-tokens 100 --temperature 0.8 --seed 42
"""

import argparse
import logging
import sys

from src.cli import resolve_source_checkpoint
from src.inference.cli import add_inference_args, build_inference_config
from src.inference.completion import complete_prompt
from src.logging_setup import setup_logging
from src.training.checkpoint import restore_from_checkpoint

logger = logging.getLogger(__name__)


def _terminal_printout(completion: str) -> None:
    """Print the completion set apart from the preceding logs by banner lines."""
    separator = "-" * 30
    print(f"\n\n{separator}\n GENERATED TEXT\n{separator}\n\n{completion}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a text completion from nanoLLM.")
    add_inference_args(parser)
    args = parser.parse_args()

    # Reject an unusable prompt here so the user is not charged the cost of restoring the model first.
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    return args


def main() -> None:
    setup_logging()
    args = _parse_args()

    # --- Prepare Inputs ---

    try:
        # Construct config from CLI arguments
        inference_config = build_inference_config(args)
    except Exception as e:
        logger.error(f"Failed to construct configs from CLI arguments: {e}")
        sys.exit(1)

    try:
        # Resolve checkpoint source bundle (defaults to most recent)
        checkpoint_source = resolve_source_checkpoint(args)
    except Exception as e:
        logger.error(f"Failed to resolve checkpoint source bundle: {e}")
        sys.exit(1)

    # --- Execute Operations ---

    try:
        model, tokenizer_config, _ = restore_from_checkpoint(checkpoint_source)
        completion = complete_prompt(
            model=model,
            tokenizer_config=tokenizer_config,
            inference_config=inference_config,
            prompt=args.prompt,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

    # Print out completion
    _terminal_printout(completion)


if __name__ == "__main__":
    main()
