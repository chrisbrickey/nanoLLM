"""
nanoLLM/src/checkpoint.py

Save and load model checkpoints. Used by training (save/resume) and inference (load weights).
"""

import logging
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx

logger = logging.getLogger(__name__)
import orbax.checkpoint as ocp

from src.paths import CHECKPOINTS_DIR, validate_project_path


def default_checkpoint_path(model_name: str = "NanoLLM") -> Path:
    """Return a timestamped checkpoint path under CHECKPOINTS_DIR."""
    return CHECKPOINTS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.orbax"


def save_checkpoint(model: nnx.Module, path: Path, *, force: bool = True) -> None:
    """Save model state to an orbax checkpoint."""
    validated_path = validate_project_path(path)
    try:
        validated_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create checkpoint directory '{validated_path.parent}': {e}") from e
    logger.info("Saving checkpoint to %s", validated_path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(validated_path, nnx.state(model), force=force)
    logger.info("Checkpoint saved.")


def get_latest_checkpoint(directory: Path = CHECKPOINTS_DIR) -> Path | None:
    """Return the most recently modified .orbax checkpoint in directory, or None."""
    if not directory.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in directory.glob("*.orbax"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    return max(candidates, key=lambda t: t[0])[1] if candidates else None


def load_checkpoint(model: nnx.Module, path: Path) -> nnx.Module:
    """Restore model state from an orbax checkpoint. Returns the updated model.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If the underlying restore fails (corrupt file, structure
            mismatch, etc.). Orbax raises a mix of error types depending on the
            failure mode; presenting one consistent ValueError gives callers a
            single error path.
    """
    validated_path = validate_project_path(path)
    if not validated_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {validated_path}")
    logger.info("Loading checkpoint from %s", validated_path)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored_state = checkpointer.restore(validated_path, item=nnx.state(model))
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise ValueError(f"Failed to load checkpoint at {validated_path}: {e}") from e
    nnx.update(model, restored_state)
    logger.info("Checkpoint loaded.")
    return model
