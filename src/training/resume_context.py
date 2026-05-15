"""
nanoLLM/src/training/resume_context.py

Joins information required for resuming training from a checkpoint
so that callers cannot pass one element without the other elements,
which could result in (for example) wrong calculation of total epochs.
"""

from dataclasses import dataclass
from pathlib import Path

from src.checkpoint import load_metadata


@dataclass(frozen=True)
class ResumeContext:
    """Resume-time context for a training run loaded from a checkpoint."""

    source: Path
    previous_epochs_completed: int

    @classmethod
    def from_checkpoint(cls, source: Path) -> "ResumeContext":
        """Build a ResumeContext by reading cumulative_epochs_completed from
        the checkpoint's metadata.

        Raises:
            ValueError: If the checkpoint has no readable metadata.
        """
        metadata = load_metadata(source)
        if metadata is None:
            raise ValueError(
                f"Cannot resume from {source}: no readable metadata found."
            )
        return cls(
            source=source,
            previous_epochs_completed=metadata.cumulative_epochs_completed,
        )
