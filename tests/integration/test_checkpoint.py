"""Integration tests for src/checkpoint.py — exercises real orbax weight
serialization. Path-validation, metadata.json handling, and error branches
are unit-tested in tests/unit/test_checkpoint.py with orbax patched."""

import dataclasses
import logging
import shutil
import uuid
from collections.abc import Callable, Generator
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.checkpoint import (
    CheckpointMetadata,
    apply_checkpoint,
    build_model_from_checkpoint,
    save_checkpoint,
)
from src.config import ModelConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR

SAMPLE_TOKENIZER_CONFIG: dict[str, object] = {
    "delimiter": "<|endoftext|>",
    "name": "gpt2",
    "pad_token_id": 0,
}

SAMPLE_MODEL_CONFIG_DICT: dict[str, object] = {
    "maxlen": 4,
    "vocab_size": 50,
    "embed_dim": 12,
    "num_heads": 3,
    "feed_forward_dim": 16,
    "num_transformer_blocks": 1,
    "model_seed": 0,
}


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"integration_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestSaveLoadRoundTrip:
    def test_restored_model_params_match_original(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        original = make_tiny_model(seed=0)

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            save_checkpoint(original, project_checkpoint_path)

            # Initialize with a different seed so params start different
            restored_model = make_tiny_model(seed=99)
            apply_checkpoint(restored_model, project_checkpoint_path)

        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        rest_leaves = jax.tree_util.tree_leaves(nnx.state(restored_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, rest_leaves))
        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text
        assert "Loading checkpoint" in caplog.text
        assert "Checkpoint loaded" in caplog.text


class TestBuildModelFromCheckpoint:
    def test_returns_model_with_correct_configs(
        self, project_checkpoint_path: Path
    ) -> None:
        model_config = ModelConfig(**SAMPLE_MODEL_CONFIG_DICT)
        tokenizer_config = TokenizerConfig(**SAMPLE_TOKENIZER_CONFIG)
        original = NanoLLM(model_config)
        metadata = CheckpointMetadata(
            epochs_trained=1,
            model_config=dataclasses.asdict(model_config),
            tokenizer_config=dataclasses.asdict(tokenizer_config),
        )
        save_checkpoint(original, project_checkpoint_path, metadata=metadata)

        loaded_model, loaded_mc, loaded_tc = build_model_from_checkpoint(project_checkpoint_path)

        assert loaded_mc == model_config
        assert loaded_tc == tokenizer_config
        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        loaded_leaves = jax.tree_util.tree_leaves(nnx.state(loaded_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, loaded_leaves))
