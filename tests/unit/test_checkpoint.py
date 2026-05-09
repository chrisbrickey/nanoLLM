"""Unit tests for src/checkpoint.py"""

import dataclasses
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

from src.checkpoint import (
    CheckpointMetadata,
    apply_checkpoint,
    build_model_from_checkpoint,
    get_latest_checkpoint,
    get_latest_checkpoints,
    load_metadata,
    save_checkpoint,
)
from src.config import ModelConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1


def _make_bundle(parent: Path, name: str) -> Path:
    bundle = parent / name
    bundle.mkdir()
    (bundle / "weights.orbax").mkdir()
    return bundle


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
    """Unique checkpoint bundle path inside the project; cleaned up after the test."""
    path = CHECKPOINTS_DIR / f"unit_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestSaveCheckpoint:
    def test_calls_orbax_save_with_correct_args(self, caplog: pytest.LogCaptureFixture) -> None:
        model = _make_model()
        save_path = CHECKPOINTS_DIR / "mock_test"

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
                mock_instance = MagicMock()
                MockCheckpointer.return_value = mock_instance

                save_checkpoint(model, save_path)

                mock_instance.save.assert_called_once()
                call = mock_instance.save.call_args
                assert call.args[0] == (save_path / "weights.orbax").resolve()
                assert call.kwargs["force"] is True

        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text

    def test_writes_metadata_json_when_metadata_provided(
        self, project_checkpoint_path: Path
    ) -> None:
        model = _make_model()
        metadata = CheckpointMetadata(
            epochs_trained=3,
            final_loss=1.23,
            model_config={"embed_dim": 12},
            training_config={"epochs": 3},
        )
        save_checkpoint(model, project_checkpoint_path, metadata=metadata)
        assert (project_checkpoint_path / "metadata.json").exists()

    def test_does_not_write_metadata_json_when_no_metadata(
        self, project_checkpoint_path: Path
    ) -> None:
        model = _make_model()
        save_checkpoint(model, project_checkpoint_path)
        assert not (project_checkpoint_path / "metadata.json").exists()

    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            save_checkpoint(model, Path("/tmp/outside"))

    def test_raises_oserror_when_mkdir_fails(self) -> None:
        model = _make_model()
        some_valid_path = CHECKPOINTS_DIR / "unit_test_mkdir_fail"
        with patch("pathlib.Path.mkdir", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="Failed to create checkpoint directory"):
                save_checkpoint(model, some_valid_path)


SAMPLE_TOKENIZER_CONFIG: dict[str, object] = {
    "delimiter": "<|endoftext|>",
    "name": "gpt2",
    "pad_token_id": 0,
}

SAMPLE_MODEL_CONFIG_DICT: dict[str, object] = {
    "maxlen": MAXLEN,
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "feed_forward_dim": FF_DIM,
    "num_transformer_blocks": NUM_BLOCKS,
    "model_seed": 0,
}


class TestCheckpointMetadata:
    def test_tokenizer_config_round_trips_through_json(
        self, project_checkpoint_path: Path
    ) -> None:
        """CheckpointMetadata persists tokenizer_config and load_metadata returns the same dict."""
        model = _make_model()
        metadata = CheckpointMetadata(
            epochs_trained=1,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        )
        save_checkpoint(model, project_checkpoint_path, metadata=metadata)

        loaded = load_metadata(project_checkpoint_path)

        assert loaded is not None
        assert loaded.tokenizer_config == SAMPLE_TOKENIZER_CONFIG

    def test_load_metadata_backward_compat_missing_tokenizer_config(
        self, tmp_path: Path
    ) -> None:
        """Old checkpoints that lack tokenizer_config load successfully with tokenizer_config=None."""
        bundle_dir = tmp_path / "old_checkpoint"
        bundle_dir.mkdir()
        # Simulate a checkpoint written before tokenizer_config was added
        (bundle_dir / "metadata.json").write_text(
            '{"epochs_trained": 3, "final_loss": 1.5, "model_config": null, '
            '"training_config": null, "created_at": "2026-01-01T00:00:00"}',
            encoding="utf-8",
        )

        result = load_metadata(bundle_dir)

        assert result is not None
        assert result.tokenizer_config is None

    def test_load_metadata_round_trip(self, project_checkpoint_path: Path) -> None:
        model = _make_model()
        epochs_trained = 5
        final_loss = 0.87
        model_config = {"embed_dim": 12, "num_heads": 3}
        training_config = {"epochs": 5, "batch_size": 4}
        metadata = CheckpointMetadata(
            epochs_trained=epochs_trained,
            final_loss=final_loss,
            model_config=model_config,
            training_config=training_config,
        )
        save_checkpoint(model, project_checkpoint_path, metadata=metadata)

        loaded = load_metadata(project_checkpoint_path)

        assert loaded is not None
        assert loaded.epochs_trained == epochs_trained
        assert loaded.final_loss == pytest.approx(final_loss)
        assert loaded.model_config == model_config
        assert loaded.training_config == training_config
        assert isinstance(loaded.created_at, str)
        assert len(loaded.created_at) > 0

    def test_load_metadata_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "empty_bundle"
        bundle_dir.mkdir()
        # No metadata.json inside
        result = load_metadata(bundle_dir)
        assert result is None

    def test_load_metadata_returns_none_for_malformed_json(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "malformed_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "metadata.json").write_text("{ this is not valid json }", encoding="utf-8")
        result = load_metadata(bundle_dir)
        assert result is None

    def test_load_metadata_returns_none_for_wrong_structure(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "wrong_structure_bundle"
        bundle_dir.mkdir()
        # Valid JSON but missing required `epochs_trained` key
        (bundle_dir / "metadata.json").write_text('{"final_loss": 0.5}', encoding="utf-8")
        result = load_metadata(bundle_dir)
        assert result is None


class TestGetLatestCheckpoint:
    def test_returns_none_when_directory_does_not_exist(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no_such_dir"
        assert get_latest_checkpoint(nonexistent) is None

    def test_returns_none_when_directory_is_empty(self, tmp_path: Path) -> None:
        assert get_latest_checkpoint(tmp_path) is None

    def test_returns_none_when_no_valid_checkpoints(self, tmp_path: Path) -> None:
        # A plain file (not a valid checkpoint bundle) should be ignored
        (tmp_path / "some_file.txt").write_text("data")
        assert get_latest_checkpoint(tmp_path) is None

    def test_returns_single_file(self, tmp_path: Path) -> None:
        bundle = _make_bundle(tmp_path, "model_001")
        assert get_latest_checkpoint(tmp_path) == bundle

    def test_returns_most_recently_modified_file(self, tmp_path: Path) -> None:
        older = _make_bundle(tmp_path, "model_old")
        newer = _make_bundle(tmp_path, "model_new")
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        assert get_latest_checkpoint(tmp_path) == newer

    def test_ignores_directories_without_weights_orbax(self, tmp_path: Path) -> None:
        valid_bundle = _make_bundle(tmp_path, "model_valid")
        invalid_bundle = tmp_path / "model_no_weights"
        invalid_bundle.mkdir()
        assert get_latest_checkpoint(tmp_path) == valid_bundle

    def test_skips_file_with_oserror_on_stat(self, tmp_path: Path) -> None:
        good_bundle = _make_bundle(tmp_path, "model_good")
        bad_bundle = _make_bundle(tmp_path, "model_bad")

        original_stat = Path.stat

        def patched_stat(self: Path, **kwargs: object) -> object:
            if self == bad_bundle:
                raise OSError("permission denied")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", patched_stat):
            result = get_latest_checkpoint(tmp_path)

        assert result == good_bundle


class TestApplyCheckpoint:
    def test_rejects_path_outside_project(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="outside the project root"):
            apply_checkpoint(model, Path("/tmp/outside"))

    def test_rejects_missing_file(self) -> None:
        model = _make_model()
        missing_path = CHECKPOINTS_DIR / f"does_not_exist_{uuid.uuid4().hex[:8]}"
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            apply_checkpoint(model, missing_path)

    def test_rejects_bundle_without_weights(self, project_checkpoint_path: Path) -> None:
        """Bundle dir exists but weights.orbax/ subdir is absent."""
        project_checkpoint_path.mkdir(parents=True, exist_ok=True)
        model = _make_model()
        with pytest.raises(FileNotFoundError, match="No weights found"):
            apply_checkpoint(model, project_checkpoint_path)

    def test_wraps_orbax_errors_as_value_error(self, project_checkpoint_path: Path) -> None:
        """Underlying orbax exceptions surface as a single ValueError so callers have one error path."""
        model = _make_model()
        # Create the bundle dir AND weights.orbax subdir so existence checks pass
        project_checkpoint_path.mkdir(parents=True, exist_ok=True)
        (project_checkpoint_path / "weights.orbax").mkdir()
        with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
            mock_instance = MagicMock()
            mock_instance.restore.side_effect = KeyError("missing tree node")
            MockCheckpointer.return_value = mock_instance
            with pytest.raises(ValueError, match="Failed to load checkpoint"):
                apply_checkpoint(model, project_checkpoint_path)


class TestSaveLoadRoundTrip:
    def test_restored_model_params_match_original(
        self, project_checkpoint_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        original = _make_model(seed=0)

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            save_checkpoint(original, project_checkpoint_path)

            # Initialize with a different seed so params start different
            restored_model = _make_model(seed=99)
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

    def test_raises_when_no_metadata(self, project_checkpoint_path: Path) -> None:
        save_checkpoint(_make_model(), project_checkpoint_path)
        with pytest.raises(ValueError, match="no metadata"):
            build_model_from_checkpoint(project_checkpoint_path)

    def test_raises_when_model_config_missing(
        self, project_checkpoint_path: Path
    ) -> None:
        metadata = CheckpointMetadata(
            epochs_trained=1, tokenizer_config=SAMPLE_TOKENIZER_CONFIG
        )
        save_checkpoint(_make_model(), project_checkpoint_path, metadata=metadata)
        with pytest.raises(ValueError, match="model_config"):
            build_model_from_checkpoint(project_checkpoint_path)

    def test_raises_when_tokenizer_config_missing(
        self, project_checkpoint_path: Path
    ) -> None:
        metadata = CheckpointMetadata(
            epochs_trained=1, model_config=SAMPLE_MODEL_CONFIG_DICT
        )
        save_checkpoint(_make_model(), project_checkpoint_path, metadata=metadata)
        with pytest.raises(ValueError, match="tokenizer_config"):
            build_model_from_checkpoint(project_checkpoint_path)

    def test_rejects_path_outside_project(self) -> None:
        with pytest.raises(ValueError, match="outside the project root"):
            build_model_from_checkpoint(Path("/tmp/outside"))


class TestGetLatestCheckpoints:
    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        result = get_latest_checkpoints(tmp_path, n=2)
        assert result == []

    def test_single_bundle_returns_list_of_length_one(self, tmp_path: Path) -> None:
        bundle = _make_bundle(tmp_path, "checkpoint_001")
        result = get_latest_checkpoints(tmp_path, n=2)
        assert result == [bundle]

    def test_three_bundles_with_n2_returns_two_newest_first(self, tmp_path: Path) -> None:
        oldest = _make_bundle(tmp_path, "checkpoint_old")
        middle = _make_bundle(tmp_path, "checkpoint_mid")
        newest = _make_bundle(tmp_path, "checkpoint_new")
        os.utime(oldest, (1_000_000, 1_000_000))
        os.utime(middle, (2_000_000, 2_000_000))
        os.utime(newest, (3_000_000, 3_000_000))

        result = get_latest_checkpoints(tmp_path, n=2)

        assert result == [newest, middle]

    def test_n_larger_than_count_returns_all_available(self, tmp_path: Path) -> None:
        bundle_a = _make_bundle(tmp_path, "checkpoint_a")
        bundle_b = _make_bundle(tmp_path, "checkpoint_b")
        os.utime(bundle_a, (1_000_000, 1_000_000))
        os.utime(bundle_b, (2_000_000, 2_000_000))

        result = get_latest_checkpoints(tmp_path, n=10)

        assert len(result) == 2
        assert set(result) == {bundle_a, bundle_b}
