"""
nanoLLM/src/training/schema.py

Value objects and data schemas for the training pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.checkpoint import load_metadata


@dataclass
class MetricsHistory:
    """Accumulated training metrics recorded at log_every_n_steps intervals."""

    train_loss: list[float] = field(default_factory=list)

    def record(self, metric_name: str, value: float) -> None:
        """Append *value* to the field for the named metric (e.g. "loss" → train_loss).

        Raises AttributeError for unknown metric names, keeping the schema explicit.
        """
        getattr(self, f"train_{metric_name}").append(value)

    @property
    def final_train_loss(self) -> float | None:
        return self.train_loss[-1] if self.train_loss else None


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
