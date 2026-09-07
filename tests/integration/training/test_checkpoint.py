"""Integration tests for src/training/checkpoint.py — exercises real orbax weight
serialization. Path-validation, metadata.json handling, and error branches
are unit-tested in tests/unit/training/test_checkpoint.py with orbax patched."""

import dataclasses
import logging
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.config import ModelConfig
from src.model.model import NanoLLM
from src.training.checkpoint import (
    apply_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)
from src.training.schema import CheckpointMetadata
from tests.conftest import SAMPLE_TOKENIZER_CONFIG, MakeTinyModel


class TestSaveLoadRoundTrip:
    def test_restored_model_params_match_original(
        self,
        make_tiny_model: MakeTinyModel,
        checkpoint_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        original = make_tiny_model(seed=0)

        with caplog.at_level(logging.INFO, logger="src.training.checkpoint"):
            save_checkpoint(original, checkpoint_path)

            # Initialize with a different seed so params start different
            restored_model = make_tiny_model(seed=99)
            apply_checkpoint(restored_model, checkpoint_path)

        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        rest_leaves = jax.tree_util.tree_leaves(nnx.state(restored_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, rest_leaves))
        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text
        assert "Loading checkpoint" in caplog.text
        assert "Checkpoint loaded" in caplog.text

    def test_restore_emits_no_sharding_warning(
        self,
        make_tiny_model: MakeTinyModel,
        checkpoint_path: Path,
    ) -> None:
        original = make_tiny_model(seed=0)
        save_checkpoint(original, checkpoint_path)
        restored_model = make_tiny_model(seed=99)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            apply_checkpoint(restored_model, checkpoint_path)

        sharding_warnings = [w for w in recorded if "Sharding info not provided" in str(w.message)]
        assert sharding_warnings == []


class TestBuildModelFromCheckpoint:
    def test_returns_model_with_correct_configs(
        self, tiny_model_config: ModelConfig, checkpoint_path: Path
    ) -> None:
        model_config = tiny_model_config
        tokenizer_config = SAMPLE_TOKENIZER_CONFIG
        original = NanoLLM(model_config)
        cumulative_epochs = 1
        metadata = CheckpointMetadata(
            cumulative_epochs_completed=cumulative_epochs,
            model_config=dataclasses.asdict(model_config),
            tokenizer_config=dataclasses.asdict(tokenizer_config),
        )
        save_checkpoint(original, checkpoint_path, metadata=metadata)

        loaded_model, loaded_tokenizer_config, loaded_metadata = restore_from_checkpoint(
            checkpoint_path
        )

        assert loaded_model.config == model_config
        assert loaded_tokenizer_config == tokenizer_config
        assert loaded_metadata.cumulative_epochs_completed == cumulative_epochs
        assert loaded_metadata.model_config == dataclasses.asdict(model_config)
        assert loaded_metadata.tokenizer_config == dataclasses.asdict(tokenizer_config)
        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        loaded_leaves = jax.tree_util.tree_leaves(nnx.state(loaded_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, loaded_leaves))
