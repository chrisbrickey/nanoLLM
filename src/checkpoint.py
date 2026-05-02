"""
nanoLLM/src/checkpoint.py

Save and load model checkpoints. Used by training (save/resume) and inference (load weights).
"""

from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp

from src.config import validate_project_path


def save_checkpoint(model: nnx.Module, path: Path, *, force: bool = True) -> None:
    """Save model state to an orbax checkpoint."""
    validated_path = validate_project_path(path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(validated_path, nnx.state(model), force=force)


def load_checkpoint(model: nnx.Module, path: Path) -> nnx.Module:
    """Restore model state from an orbax checkpoint. Returns the updated model."""
    validated_path = validate_project_path(path)
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(validated_path, item=nnx.state(model))
    nnx.update(model, restored_state)
    return model
