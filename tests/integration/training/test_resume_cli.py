"""Integration tests for the resume CLI entry point (scripts/resume.py)."""

import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.resume import main as resume_main
from scripts.train import main as train_main
from src.paths import CHECKPOINTS_DIR
from src.training.checkpoint import load_metadata
from tests.conftest import CheckpointPathFactory


class TestResumeCliHappyPath:
    def test_train_then_resume_doubles_cumulative_epochs(
        self, data_file: Path, checkpoint_path_factory: CheckpointPathFactory
    ) -> None:
        """End-to-end: train one epoch, then resume one more epoch. The
        resulting checkpoint's metadata must record cumulative_epochs_completed
        equal to the sum of both phases."""
        first_path = checkpoint_path_factory("resume_cli_test_first")
        second_path = checkpoint_path_factory("resume_cli_test_second")
        epochs_per_phase = 1

        train_argv = [
            "nanollm-train",
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", str(epochs_per_phase),
            "--batch-size", "2",
            "--checkpoint-destination", str(first_path),
        ]
        with patch("sys.argv", train_argv):
            train_main()
        assert first_path.exists()

        resume_argv = [
            "nanollm-resume",
            "--checkpoint-source", str(first_path),
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", str(epochs_per_phase),
            "--batch-size", "2",
            "--checkpoint-destination", str(second_path),
        ]
        with patch("sys.argv", resume_argv):
            resume_main()

        assert second_path.exists()
        assert (second_path / "weights.orbax").exists()
        metadata = load_metadata(second_path)
        assert metadata is not None
        assert metadata.cumulative_epochs_completed == epochs_per_phase * 2


class TestResumeCliSourceCheckpointResolution:
    """Verifies that --checkpoint-source and its fallback (get_latest_checkpoint)
    flow correctly into the Runner's checkpoint_source kwarg. Trainer
    execution is patched so these tests stay fast and don't write real bundles."""

    @pytest.fixture
    def patched_run(self, data_file: Path):
        """Patches downstream training so only the source-resolution path is exercised."""

        def _run(argv: list[str], *, latest: Path | None = None) -> MagicMock:
            with patch("scripts.resume.get_latest_checkpoint", return_value=latest), \
                 patch("scripts.resume.Runner") as mock_runner_cls:
                mock_runner_cls.return_value.run.return_value = None
                with patch("sys.argv", argv):
                    resume_main()
                return mock_runner_cls

        return _run

    def test_explicit_source_checkpoint_flag_forwarded(
        self, patched_run, data_file: Path
    ) -> None:
        explicit_source = CHECKPOINTS_DIR / "explicit_source.orbax"
        argv = [
            "nanollm-resume",
            "--checkpoint-source", str(explicit_source),
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
        ]
        mock_runner_cls = patched_run(argv)
        assert mock_runner_cls.call_args.kwargs["checkpoint_source"] == explicit_source

    def test_falls_back_to_latest_when_source_omitted(
        self, patched_run, data_file: Path
    ) -> None:
        latest = CHECKPOINTS_DIR / "latest_auto.orbax"
        argv = [
            "nanollm-resume",
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
        ]
        mock_runner_cls = patched_run(argv, latest=latest)
        assert mock_runner_cls.call_args.kwargs["checkpoint_source"] == latest


class TestResumeCliErrors:
    @pytest.fixture(autouse=True)
    def _capture_logs(self, caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
        with caplog.at_level(logging.ERROR, logger="scripts.resume"):
            yield

    def test_no_source_and_no_checkpoints_exits_1(
        self, caplog: pytest.LogCaptureFixture, data_file: Path
    ) -> None:
        """When --checkpoint-source is omitted and no bundles exist, the CLI
        must exit 1 and log a clear error mentioning that no checkpoints were found."""
        argv = [
            "nanollm-resume",
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
        ]
        with patch("scripts.resume.get_latest_checkpoint", return_value=None):
            with patch("sys.argv", argv):
                with pytest.raises(SystemExit) as exc_info:
                    resume_main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert "No checkpoints found" in caplog.text

    def test_nonexistent_source_checkpoint_exits_1(
        self, caplog: pytest.LogCaptureFixture, data_file: Path
    ) -> None:
        """If --checkpoint-source points at a missing bundle, the CLI must exit 1."""
        missing = CHECKPOINTS_DIR / "nonexistent_bundle"
        argv = [
            "nanollm-resume",
            "--checkpoint-source", str(missing),
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                resume_main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)
