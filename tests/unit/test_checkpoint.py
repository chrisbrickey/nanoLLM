"""Unit tests for src/checkpoint.py"""

import logging
import os
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.checkpoint import get_latest_checkpoint, load_checkpoint, save_checkpoint
from src.config import ModelConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1


def _make_model(seed: int = 0) -> NanoLLM:
    config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        model_seed=seed,
    )
    return NanoLLM(config)


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    """Unique checkpoint path inside the project; cleaned up after the test."""
    path = CHECKPOINTS_DIR / f"unit_test_{uuid.uuid4().hex[:8]}.orbax"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestSaveCheckpoint:
    def test_calls_orbax_save_with_correct_args(self, caplog: pytest.LogCaptureFixture) -> None:
        model = _make_model()
        save_path = CHECKPOINTS_DIR / "mock_test.orbax"

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
                mock_instance = MagicMock()
                MockCheckpointer.return_value = mock_instance

                save_checkpoint(model, save_path)

                mock_instance.save.assert_called_once()
                call = mock_instance.save.call_args
                assert call.args[0] == save_path.resolve()
                assert call.kwargs["force"] is True

        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text

    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            save_checkpoint(model, Path("/tmp/outside.orbax"))

    def test_raises_oserror_when_mkdir_fails(self) -> None:
        model = _make_model()
        some_valid_path = CHECKPOINTS_DIR / "unit_test_mkdir_fail.orbax"
        with patch("pathlib.Path.mkdir", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="Failed to create checkpoint directory"):
                save_checkpoint(model, some_valid_path)


class TestGetLatestCheckpoint:
    def test_returns_none_when_directory_does_not_exist(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no_such_dir"
        assert get_latest_checkpoint(nonexistent) is None

    def test_returns_none_when_directory_is_empty(self, tmp_path: Path) -> None:
        assert get_latest_checkpoint(tmp_path) is None

    def test_returns_none_when_no_orbax_files(self, tmp_path: Path) -> None:
        (tmp_path / "some_file.txt").write_text("data")
        assert get_latest_checkpoint(tmp_path) is None

    def test_returns_single_file(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "model_001.orbax"
        checkpoint.write_text("checkpoint data")
        assert get_latest_checkpoint(tmp_path) == checkpoint

    def test_returns_most_recently_modified_file(self, tmp_path: Path) -> None:
        older = tmp_path / "model_old.orbax"
        newer = tmp_path / "model_new.orbax"
        older.write_text("old data")
        newer.write_text("new data")
        # Set older mtime to the past explicitly
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        assert get_latest_checkpoint(tmp_path) == newer

    def test_ignores_non_orbax_files(self, tmp_path: Path) -> None:
        orbax_file = tmp_path / "model_001.orbax"
        text_file = tmp_path / "notes.txt"
        orbax_file.write_text("checkpoint data")
        text_file.write_text("some notes")
        assert get_latest_checkpoint(tmp_path) == orbax_file

    def test_skips_file_with_oserror_on_stat(self, tmp_path: Path) -> None:
        good_file = tmp_path / "model_good.orbax"
        bad_file = tmp_path / "model_bad.orbax"
        good_file.write_text("good checkpoint")
        bad_file.write_text("bad checkpoint")

        original_stat = Path.stat

        def patched_stat(self: Path, **kwargs: object) -> object:
            if self == bad_file:
                raise OSError("permission denied")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", patched_stat):
            result = get_latest_checkpoint(tmp_path)

        assert result == good_file


class TestLoadCheckpoint:
    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            load_checkpoint(model, Path("/tmp/outside.orbax"))


class TestSaveLoadRoundTrip:
    def test_restored_model_params_match_original(
        self, project_checkpoint_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        original = _make_model(seed=0)

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            save_checkpoint(original, project_checkpoint_path)

            # Initialize with a different seed so params start different
            restored_model = _make_model(seed=99)
            load_checkpoint(restored_model, project_checkpoint_path)

        orig_leaves = jax.tree_util.tree_leaves(nnx.state(original))
        rest_leaves = jax.tree_util.tree_leaves(nnx.state(restored_model))
        assert all(jnp.allclose(a, b) for a, b in zip(orig_leaves, rest_leaves))
        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text
        assert "Loading checkpoint" in caplog.text
        assert "Checkpoint loaded" in caplog.text
