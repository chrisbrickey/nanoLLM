"""Unit tests for src/checkpoint.py — orbax I/O is patched everywhere
save_checkpoint is exercised, so these tests verify path validation,
metadata.json read/write branches, and error handling without doing real
weight serialization. Save→load round-trip with real orbax lives in
tests/integration/test_checkpoint.py."""

import dataclasses
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from src.config import TokenizerConfig
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


def _make_bundle(parent: Path, name: str) -> Path:
    bundle = parent / name
    bundle.mkdir()
    (bundle / "weights.orbax").mkdir()
    return bundle


def _write_metadata_json(bundle_dir: Path, payload: dict) -> Path:
    """Write a metadata.json directly to bundle_dir, bypassing save_checkpoint
    and orbax. Use when the test only exercises load_metadata."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = bundle_dir / "metadata.json"
    metadata_file.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_file


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    """Unique checkpoint bundle path inside the project; cleaned up after the
    test. Required only when calling save_checkpoint, since save_checkpoint
    runs validate_project_path which rejects paths outside the project root."""
    path = CHECKPOINTS_DIR / f"unit_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def patched_orbax() -> Generator[MagicMock, None, None]:
    """Patches ocp.PyTreeCheckpointer so save_checkpoint does no real weight
    I/O. Yields the mock instance for tests that need to assert on it."""
    with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
        instance = MagicMock()
        MockCheckpointer.return_value = instance
        yield instance


class TestSaveCheckpoint:
    def test_calls_orbax_save_with_correct_args(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        patched_orbax: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            save_checkpoint(make_tiny_model(), project_checkpoint_path)

        patched_orbax.save.assert_called_once()
        call = patched_orbax.save.call_args
        assert call.args[0] == (project_checkpoint_path / "weights.orbax").resolve()
        assert call.kwargs["force"] is True
        assert "Saving checkpoint" in caplog.text
        assert "Checkpoint saved" in caplog.text

    def test_writes_metadata_json_when_metadata_provided(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        patched_orbax: MagicMock,
    ) -> None:
        metadata = CheckpointMetadata(
            cumulative_epochs_completed=3,
            final_loss=1.23,
            model_config={"embed_dim": 12},
            training_config={"epochs": 3},
        )
        save_checkpoint(make_tiny_model(), project_checkpoint_path, metadata=metadata)
        assert (project_checkpoint_path / "metadata.json").exists()

    def test_does_not_write_metadata_json_when_no_metadata(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        patched_orbax: MagicMock,
    ) -> None:
        save_checkpoint(make_tiny_model(), project_checkpoint_path)
        assert not (project_checkpoint_path / "metadata.json").exists()

    def test_rejects_path_outside_project(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        with pytest.raises(ValueError, match="outside the project root"):
            save_checkpoint(make_tiny_model(), Path("/tmp/outside"))

    def test_raises_oserror_when_mkdir_fails(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        some_valid_path = CHECKPOINTS_DIR / "unit_test_mkdir_fail"
        with patch("pathlib.Path.mkdir", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="Failed to create checkpoint directory"):
                save_checkpoint(make_tiny_model(), some_valid_path)


class TestCheckpointMetadata:
    def test_tokenizer_config_round_trips_through_json(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        patched_orbax: MagicMock,
    ) -> None:
        """save_checkpoint persists tokenizer_config in metadata.json and
        load_metadata returns the same dict — exercised without orbax."""
        metadata = CheckpointMetadata(
            cumulative_epochs_completed=1,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        )
        save_checkpoint(make_tiny_model(), project_checkpoint_path, metadata=metadata)

        loaded = load_metadata(project_checkpoint_path)

        assert loaded is not None
        assert loaded.tokenizer_config == SAMPLE_TOKENIZER_CONFIG

    def test_load_metadata_backward_compat_missing_tokenizer_config(
        self, tmp_path: Path
    ) -> None:
        """Old checkpoints that lack tokenizer_config load with tokenizer_config=None."""
        bundle_dir = tmp_path / "old_checkpoint"
        _write_metadata_json(
            bundle_dir,
            {
                "cumulative_epochs_completed": 3,
                "final_loss": 1.5,
                "model_config": None,
                "training_config": None,
                "created_at": "2026-01-01T00:00:00",
            },
        )

        result = load_metadata(bundle_dir)

        assert result is not None
        assert result.tokenizer_config is None

    def test_load_metadata_round_trip(
        self,
        make_tiny_model: Callable[..., NanoLLM],
        project_checkpoint_path: Path,
        patched_orbax: MagicMock,
    ) -> None:
        cumulative_epochs_completed = 5
        final_loss = 0.87
        model_config = {"embed_dim": 12, "num_heads": 3}
        training_config = {"epochs": 5, "batch_size": 4}
        metadata = CheckpointMetadata(
            cumulative_epochs_completed=cumulative_epochs_completed,
            final_loss=final_loss,
            model_config=model_config,
            training_config=training_config,
        )
        save_checkpoint(make_tiny_model(), project_checkpoint_path, metadata=metadata)

        loaded = load_metadata(project_checkpoint_path)

        assert loaded is not None
        assert loaded.cumulative_epochs_completed == cumulative_epochs_completed
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
        _write_metadata_json(bundle_dir, {"final_loss": 0.5})  # missing cumulative_epochs_completed
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
    def test_rejects_path_outside_project(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        with pytest.raises(ValueError, match="outside the project root"):
            apply_checkpoint(make_tiny_model(), Path("/tmp/outside"))

    def test_rejects_missing_file(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        missing_path = CHECKPOINTS_DIR / f"does_not_exist_{uuid.uuid4().hex[:8]}"
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            apply_checkpoint(make_tiny_model(), missing_path)

    def test_rejects_bundle_without_weights(
        self, make_tiny_model: Callable[..., NanoLLM], project_checkpoint_path: Path
    ) -> None:
        """Bundle dir exists but weights.orbax/ subdir is absent."""
        project_checkpoint_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="No weights found"):
            apply_checkpoint(make_tiny_model(), project_checkpoint_path)

    def test_wraps_orbax_errors_as_value_error(
        self, make_tiny_model: Callable[..., NanoLLM], project_checkpoint_path: Path
    ) -> None:
        """Underlying orbax exceptions surface as a single ValueError so callers have one error path."""
        project_checkpoint_path.mkdir(parents=True, exist_ok=True)
        (project_checkpoint_path / "weights.orbax").mkdir()
        with patch("src.checkpoint.ocp.PyTreeCheckpointer") as MockCheckpointer:
            mock_instance = MagicMock()
            mock_instance.restore.side_effect = KeyError("missing tree node")
            MockCheckpointer.return_value = mock_instance
            with pytest.raises(ValueError, match="Failed to load checkpoint"):
                apply_checkpoint(make_tiny_model(), project_checkpoint_path)


class TestBuildModelFromCheckpoint:
    """Unit-level tests of build_model_from_checkpoint's metadata-validation
    branches. The happy-path case (full reconstruction with real weights) is
    in tests/integration/test_checkpoint.py."""

    def test_raises_when_no_metadata(self, project_checkpoint_path: Path) -> None:
        # Bundle has weights.orbax (passes apply_checkpoint's existence check)
        # but no metadata.json — build_model_from_checkpoint must reject early.
        _make_bundle(project_checkpoint_path.parent, project_checkpoint_path.name)
        with pytest.raises(ValueError, match="no metadata"):
            build_model_from_checkpoint(project_checkpoint_path)

    def test_raises_when_model_config_missing(
        self, project_checkpoint_path: Path
    ) -> None:
        _write_metadata_json(
            project_checkpoint_path,
            {
                "cumulative_epochs_completed": 1,
                "tokenizer_config": SAMPLE_TOKENIZER_CONFIG,
            },
        )
        with pytest.raises(ValueError, match="model_config"):
            build_model_from_checkpoint(project_checkpoint_path)

    def test_raises_when_tokenizer_config_missing(
        self, project_checkpoint_path: Path
    ) -> None:
        _write_metadata_json(
            project_checkpoint_path,
            {
                "cumulative_epochs_completed": 1,
                "model_config": SAMPLE_MODEL_CONFIG_DICT,
            },
        )
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
