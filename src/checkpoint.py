"""
nanoLLM/src/checkpoint.py

Save and load model checkpoints. Used by training (save/resume) and inference (load weights).
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import flax.nnx as nnx

logger = logging.getLogger(__name__)
import orbax.checkpoint as ocp

from src.config import ModelConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR, validate_project_path


@dataclass
class CheckpointMetadata:
    epochs_trained: int
    final_loss: float | None = None
    model_config: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tokenizer_config: dict[str, Any] | None = None

#--- save checkpoint ---

def default_checkpoint_path(model_name: str = "NanoLLM") -> Path:
    """Return a timestamped checkpoint path under CHECKPOINTS_DIR."""
    return CHECKPOINTS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_checkpoint(
    model: nnx.Module,
    path: Path,
    *,
    metadata: CheckpointMetadata | None = None,
    force: bool = True,
) -> None:
    """Save model state to an orbax checkpoint bundle."""
    validated_path = validate_project_path(path)
    try:
        validated_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create checkpoint directory '{validated_path}': {e}") from e
    logger.info("Saving checkpoint to %s", validated_path)
    checkpointer = ocp.PyTreeCheckpointer()
    weights_path = validated_path / "weights.orbax"
    checkpointer.save(weights_path.resolve(), nnx.state(model), force=force)
    if metadata is not None:
        _write_metadata(validated_path, metadata)
    logger.info("Checkpoint saved.")

def _write_metadata(bundle_path: Path, metadata: CheckpointMetadata) -> None:
    (bundle_path / "metadata.json").write_text(
        json.dumps(dataclasses.asdict(metadata), indent=2),
        encoding="utf-8",
    )

#--- load checkpoint ---

def get_latest_checkpoint(directory: Path = CHECKPOINTS_DIR) -> Path | None:
    """Return the most recently modified checkpoint bundle in directory, or None."""
    if not directory.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in directory.iterdir():
        try:
            if not (p.is_dir() and (p / "weights.orbax").exists()):
                continue
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    return max(candidates, key=lambda t: t[0])[1] if candidates else None

def load_metadata(path: Path) -> CheckpointMetadata | None:
    """Read metadata.json from a checkpoint bundle.
    This remains a public method for callers who want to inspect the metadata.

    Returns:
        CheckpointMetadata or None if not present or if the file cannot be parsed.
    """
    metadata_file = path / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
        return CheckpointMetadata(
            epochs_trained=data["epochs_trained"],
            final_loss=data.get("final_loss"),
            model_config=data.get("model_config"),
            training_config=data.get("training_config"),
            created_at=data.get("created_at", ""),
            tokenizer_config=data.get("tokenizer_config"),
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def apply_checkpoint(model: nnx.Module, path: Path) -> nnx.Module:
    """Restore weights to an existing model (restores model state) from an orbax checkpoint bundle.
    Use when model already exists, e.g., during a training session. This function
    restores the model to resume training that was previously paused.

    Returns:
         updated model

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

    weights_path = validated_path / "weights.orbax"
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights found at {weights_path}")

    logger.info("Loading checkpoint from %s", validated_path)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored_state = checkpointer.restore(weights_path, item=nnx.state(model))
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise ValueError(f"Failed to load checkpoint at {validated_path}: {e}") from e

    nnx.update(model, restored_state)
    logger.info("Checkpoint loaded.")
    return model


_MANUAL_LOAD_HINT = (
    "To load a checkpoint without complete metadata, use the manual approach: "
    "create ModelConfig() and NanoLLM(model_config), call apply_checkpoint(model, path)."
)


def build_model_from_checkpoint(
    path: Path,
) -> tuple[NanoLLM, ModelConfig, TokenizerConfig]:
    """Build a NanoLLM and reconstruct its configs from a checkpoint bundle.

    Requires a complete metadata.json in the checkpoint bundle — both model_config
    and tokenizer_config must be present. Use this when rebuilding a model from scratch.

    If metadata is absent or incomplete, use the manual approach instead:
    Call apply_checkpoint() to restore weights and manually construct the configs.

    Returns:
        Tuple of (model, model_config, tokenizer_config).

    Raises:
        FileNotFoundError: path or weights.orbax missing (delegated to apply_checkpoint).
        ValueError: metadata absent, or model_config/tokenizer_config missing from it.
    """
    validated_path = validate_project_path(path)
    logger.info("Rebuilding model from checkpoint at %s", validated_path)

    metadata = load_metadata(validated_path)
    if metadata is None:
        raise ValueError(
            f"Cannot build model: no metadata found at '{validated_path}'."
            f"{_MANUAL_LOAD_HINT}"
        )

    if metadata.model_config is None:
        raise ValueError(
            f"metadata.json at '{validated_path}' is missing model_config. "
            f"{_MANUAL_LOAD_HINT}"
        )

    if metadata.tokenizer_config is None:
        raise ValueError(
            f"metadata.json at '{validated_path}' is missing tokenizer_config. "
            f"{_MANUAL_LOAD_HINT}"
        )

    # construct relevant configs
    model_config = ModelConfig(**metadata.model_config)
    tokenizer_config = TokenizerConfig(**metadata.tokenizer_config)

    # construct and update model
    model = NanoLLM(model_config)
    apply_checkpoint(model, validated_path)

    return model, model_config, tokenizer_config
