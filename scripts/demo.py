"""
nanoLLM/scripts/demo.py

CLI entry point for launching a web demo of nanoLLM text generation.

Loads an existing checkpoint bundle and launches a Gradio UI for interactively generating completions.

Usage:
  # launch the demo using defaults (most recent checkpoint)
  uv run nanollm-demo

  # launch the demo with a specific checkpoint
  uv run nanollm-demo --checkpoint-source checkpoints/bundle_X/
"""

import argparse
import logging
import sys

from src.cli import resolve_source_checkpoint
from src.inference.demo import build_demo
from src.logging_setup import setup_logging
from src.training.checkpoint import restore_from_checkpoint

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a web demo for nanoLLM text generation.")
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        default=None,
        help="Path to checkpoint bundle to load. If not specified, the most recent checkpoint is loaded.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()

    try:
        checkpoint_source = resolve_source_checkpoint(args)
    except Exception as e:
        logger.error(f"Failed to resolve checkpoint source bundle: {e}")
        sys.exit(1)

    try:
        model, tokenizer_config, _ = restore_from_checkpoint(checkpoint_source)
        demo = build_demo(model=model, tokenizer_config=tokenizer_config)
    except Exception as e:
        logger.error(f"Failed to build demo: {e}")
        sys.exit(1)

    demo.launch()


if __name__ == "__main__":
    main()
