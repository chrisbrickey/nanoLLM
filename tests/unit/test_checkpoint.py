"""Unit tests for src/checkpoint.py"""

import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import CHECKPOINTS_DIR
from src.model.model import NanoLLM

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1


def _make_model(seed: int = 0) -> NanoLLM:
    return NanoLLM(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        rngs=nnx.Rngs(seed),
    )


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    """Unique checkpoint path inside the project; cleaned up after the test."""
    path = CHECKPOINTS_DIR / f"unit_test_{uuid.uuid4().hex[:8]}.orbax"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestSaveCheckpoint:
    def test_calls_orbax_save_with_correct_args(self) -> None:
        model = _make_model()
        save_path = CHECKPOINTS_DIR / "mock_test.orbax"

        with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
            mock_instance = MagicMock()
            MockCheckpointer.return_value = mock_instance

            save_checkpoint(model, save_path)

            mock_instance.save.assert_called_once()
            call = mock_instance.save.call_args
            assert call.args[0] == save_path.resolve()
            assert call.kwargs["force"] is True

    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            save_checkpoint(model, Path("/tmp/outside.orbax"))


class TestLoadCheckpoint:
    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            load_checkpoint(model, Path("/tmp/outside.orbax"))


class TestSaveLoadRoundTrip:
    def test_restored_model_params_match_original(
        self, project_checkpoint_path: Path
    ) -> None:
        original = _make_model(seed=0)
        save_checkpoint(original, project_checkpoint_path)

        # Initialize with a different seed so params start different
        restored_model = _make_model(seed=99)
        load_checkpoint(restored_model, project_checkpoint_path)

        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        rest_leaves = jax.tree_util.tree_leaves(nnx.state(restored_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, rest_leaves))
