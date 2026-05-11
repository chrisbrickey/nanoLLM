"""Integration tests for scripts/compare_checkpoints.py CLI."""

import dataclasses
import logging
import os
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.compare_checkpoints import main
from src.checkpoint import CheckpointMetadata, save_checkpoint
from src.compare import DEFAULT_CHANGE_THRESHOLD
from src.config import ModelConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR

# ---------------------------------------------------------------------------
# Small model constants to keep tests fast
# ---------------------------------------------------------------------------

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1


def _make_model_with_config(seed: int = 0) -> tuple[NanoLLM, ModelConfig]:
    config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        model_seed=seed,
    )
    return NanoLLM(config), config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_checkpoints() -> Generator[tuple[Path, Path], None, None]:
    """Save two real checkpoints with different seeds.

    Returns (older_path, newer_path) so callers can pass --before/--after
    or rely on mtime ordering.
    """
    prefix = f"compare_test_{uuid.uuid4().hex[:8]}"
    path_a = CHECKPOINTS_DIR / f"{prefix}_a"
    path_b = CHECKPOINTS_DIR / f"{prefix}_b"

    tokenizer_config = TokenizerConfig()

    model_a, config_a = _make_model_with_config(seed=0)
    metadata_a = CheckpointMetadata(
        cumulative_epochs_completed=1,
        model_config=dataclasses.asdict(config_a),
        tokenizer_config=dataclasses.asdict(tokenizer_config),
    )
    save_checkpoint(model_a, path_a, metadata=metadata_a)

    model_b, config_b = _make_model_with_config(seed=42)
    metadata_b = CheckpointMetadata(
        cumulative_epochs_completed=2,
        model_config=dataclasses.asdict(config_b),
        tokenizer_config=dataclasses.asdict(tokenizer_config),
    )
    save_checkpoint(model_b, path_b, metadata=metadata_b)

    # Ensure path_a is older so mtime-based ordering is deterministic
    os.utime(path_a, (1_000_000, 1_000_000))
    os.utime(path_b, (2_000_000, 2_000_000))

    yield path_a, path_b

    for p in [path_a, path_b]:
        if p.exists():
            shutil.rmtree(p)


@pytest.fixture
def one_checkpoint() -> Generator[Path, None, None]:
    """Save a single real checkpoint for error-path tests."""
    prefix = f"compare_test_{uuid.uuid4().hex[:8]}"
    path = CHECKPOINTS_DIR / f"{prefix}_only"

    tokenizer_config = TokenizerConfig()
    model, config = _make_model_with_config(seed=0)
    metadata = CheckpointMetadata(
        cumulative_epochs_completed=1,
        model_config=dataclasses.asdict(config),
        tokenizer_config=dataclasses.asdict(tokenizer_config),
    )
    save_checkpoint(model, path, metadata=metadata)

    yield path

    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def checkpoint_without_metadata() -> Generator[Path, None, None]:
    """Save a checkpoint bundle with no metadata.json."""
    prefix = f"compare_test_{uuid.uuid4().hex[:8]}"
    path = CHECKPOINTS_DIR / f"{prefix}_no_meta"

    model, _ = _make_model_with_config(seed=0)
    save_checkpoint(model, path)  # no metadata kwarg

    yield path

    if path.exists():
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# TestCliHappyPath
# ---------------------------------------------------------------------------


class TestCliHappyPath:
    def test_explicit_before_after_prints_both_reports(
        self,
        two_checkpoints: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path_a, path_b = two_checkpoints
        with patch(
            "sys.argv",
            ["nanollm-compare", "--before", str(path_a), "--after", str(path_b)],
        ):
            main()

        out = capsys.readouterr().out
        assert "WEIGHT MAGNITUDE" in out
        assert "STATE COMPARISON" in out

    def test_default_invocation_uses_two_most_recent_and_prints_both_reports(
        self,
        two_checkpoints: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No --before/--after: CLI picks the two most recent checkpoints by mtime.

        get_latest_checkpoints is mocked so this test is not affected by real
        checkpoints on disk. The mtime-ordering logic inside that function is
        covered by tests/unit/test_checkpoint.py::TestGetLatestCheckpoints.
        """
        path_a, path_b = two_checkpoints
        # newest-first matches the real return order of get_latest_checkpoints
        with patch("sys.argv", ["nanollm-compare"]):
            with patch(
                "scripts.compare_checkpoints.get_latest_checkpoints",
                return_value=[path_b, path_a],
            ):
                main()

        out = capsys.readouterr().out
        assert "WEIGHT MAGNITUDE" in out
        assert "STATE COMPARISON" in out


# ---------------------------------------------------------------------------
# TestCliErrors
# ---------------------------------------------------------------------------


class TestCliErrors:
    def test_fewer_than_two_checkpoints_exits_1(
        self,
        one_checkpoint: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When fewer than 2 checkpoints exist and no paths are given, CLI must exit 1."""
        with caplog.at_level(logging.ERROR, logger="scripts.compare_checkpoints"):
            with patch("sys.argv", ["nanollm-compare"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_only_before_without_after_exits_1(
        self,
        two_checkpoints: tuple[Path, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path_a, _ = two_checkpoints
        with caplog.at_level(logging.ERROR, logger="scripts.compare_checkpoints"):
            with patch("sys.argv", ["nanollm-compare", "--before", str(path_a)]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_only_after_without_before_exits_1(
        self,
        two_checkpoints: tuple[Path, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _, path_b = two_checkpoints
        with caplog.at_level(logging.ERROR, logger="scripts.compare_checkpoints"):
            with patch("sys.argv", ["nanollm-compare", "--after", str(path_b)]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_bundle_missing_metadata_exits_1_and_logs_path(
        self,
        checkpoint_without_metadata: Path,
        two_checkpoints: tuple[Path, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If --before refers to a bundle without metadata, CLI must exit 1 and log the path."""
        _, path_b = two_checkpoints
        with caplog.at_level(logging.ERROR, logger="scripts.compare_checkpoints"):
            with patch(
                "sys.argv",
                [
                    "nanollm-compare",
                    "--before", str(checkpoint_without_metadata),
                    "--after", str(path_b),
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        # The logged error must mention the problematic bundle path
        assert str(checkpoint_without_metadata) in caplog.text


# ---------------------------------------------------------------------------
# TestCliFlags
# ---------------------------------------------------------------------------


class TestCliFlags:
    def test_threshold_flag_propagates_to_formatted_output(
        self,
        two_checkpoints: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path_a, path_b = two_checkpoints
        custom_threshold = 1e-6
        with patch(
            "sys.argv",
            [
                "nanollm-compare",
                "--before", str(path_a),
                "--after", str(path_b),
                "--threshold", str(custom_threshold),
            ],
        ):
            main()

        out = capsys.readouterr().out
        assert "1e-06" in out or "1e-6" in out or "0.000001" in out

    def test_omitting_threshold_uses_default(
        self,
        two_checkpoints: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path_a, path_b = two_checkpoints
        with patch(
            "sys.argv",
            ["nanollm-compare", "--before", str(path_a), "--after", str(path_b)],
        ):
            main()

        out = capsys.readouterr().out
        default_str = str(DEFAULT_CHANGE_THRESHOLD)
        assert default_str in out or "1e-08" in out or "1e-8" in out

    def test_threshold_zero_exits_2(
        self,
        two_checkpoints: tuple[Path, Path],
    ) -> None:
        """--threshold 0 is invalid; argparse must exit with code 2."""
        path_a, path_b = two_checkpoints
        with patch(
            "sys.argv",
            [
                "nanollm-compare",
                "--before", str(path_a),
                "--after", str(path_b),
                "--threshold", "0",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_threshold_negative_exits_2(
        self,
        two_checkpoints: tuple[Path, Path],
    ) -> None:
        """--threshold -0.5 is invalid; argparse must exit with code 2."""
        path_a, path_b = two_checkpoints
        with patch(
            "sys.argv",
            [
                "nanollm-compare",
                "--before", str(path_a),
                "--after", str(path_b),
                "--threshold", "-0.5",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_threshold_positive_succeeds(
        self,
        two_checkpoints: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A small positive threshold must not raise SystemExit."""
        path_a, path_b = two_checkpoints
        with patch(
            "sys.argv",
            [
                "nanollm-compare",
                "--before", str(path_a),
                "--after", str(path_b),
                "--threshold", "1e-3",
            ],
        ):
            main()  # must complete without raising
