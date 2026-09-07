"""Integration tests for scripts/compare_checkpoints.py CLI."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.compare_checkpoints import main
from src.compare import DEFAULT_CHANGE_THRESHOLD
from tests.conftest import CheckpointPathFactory, SaveTinyCheckpoint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_checkpoints(
    checkpoint_path_factory: CheckpointPathFactory,
    save_tiny_checkpoint: SaveTinyCheckpoint,
) -> tuple[Path, Path]:
    """Save two real checkpoints with different seeds.

    Returns (older_path, newer_path) so callers can pass --before/--after
    or rely on mtime ordering.
    """
    path_a = save_tiny_checkpoint(
        checkpoint_path_factory("compare_test_a"), seed=0, cumulative_epochs_completed=1
    )
    path_b = save_tiny_checkpoint(
        checkpoint_path_factory("compare_test_b"), seed=42, cumulative_epochs_completed=2
    )

    # Ensure path_a is older so mtime-based ordering is deterministic
    os.utime(path_a, (1_000_000, 1_000_000))
    os.utime(path_b, (2_000_000, 2_000_000))

    return path_a, path_b


@pytest.fixture
def one_checkpoint(
    checkpoint_path_factory: CheckpointPathFactory,
    save_tiny_checkpoint: SaveTinyCheckpoint,
) -> Path:
    """Save a single real checkpoint for error-path tests."""
    return save_tiny_checkpoint(checkpoint_path_factory("compare_test_only"))


@pytest.fixture
def checkpoint_without_metadata(
    checkpoint_path_factory: CheckpointPathFactory,
    save_tiny_checkpoint: SaveTinyCheckpoint,
) -> Path:
    """Save a checkpoint bundle with no metadata.json."""
    return save_tiny_checkpoint(
        checkpoint_path_factory("compare_test_no_meta"), with_metadata=False
    )


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
