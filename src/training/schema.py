"""
nanoLLM/src/training/schema.py

Value objects and data schemas for the training pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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


@dataclass
class CheckpointMetadata:
    """Persisted alongside a checkpoint bundle's weights.

    Holds enough state to reconstruct the model/tokenizer
    from scratch (before applying the orbax weights) and
    resume training with accurate cumulative epoch count.
    """

    cumulative_epochs_completed: int
    final_loss: float | None = None
    model_config: dict[str, Any] | None = None
    tokenizer_config: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
