"""
nanoLLM/src/training/__init__.py

Training flows through four layers, from process boundary to core algorithm:

scripts/train.py or scripts/resume.py (CLI entry points)
-> cli.py (flags to typed configs)
-> runner.py (orchestration: build or restore model, load data, persist checkpoint)
-> trainer.py (the training loop: optimizer, schedule, gradient updates)
"""

from .schema import CheckpointMetadata, MetricsHistory
from .runner import Runner
from .trainer import Trainer

__all__ = ["CheckpointMetadata", "MetricsHistory", "Runner", "Trainer"]
