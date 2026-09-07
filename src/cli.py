"""Functionality shared by CLI scripts"""

import argparse
import dataclasses
from pathlib import Path
from typing import TypeVar

from src.paths import CHECKPOINTS_DIR
from src.training.checkpoint import get_latest_checkpoint

ConfigT = TypeVar("ConfigT")


def apply_cli_overrides(config: ConfigT, args: argparse.Namespace) -> ConfigT:
    """Return a copy of config with non-None CLI values replacing the defaults."""
    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(config)
        if getattr(args, field.name, None) is not None
    }
    return dataclasses.replace(config, **overrides)


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
