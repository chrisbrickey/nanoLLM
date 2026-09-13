"""Integration tests for the resume CLI entry point (scripts/resume.py)."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.resume import main as resume_main
from scripts.train import main as train_main
from src.paths import CHECKPOINTS_DIR
from src.training.checkpoint import load_metadata
from tests.conftest import (
    CheckpointPathFactory,
    MakeRunCli,
    RunCli,
    assert_error_exit,
)

RESUME_PROG = "nanollm-resume"
TRAIN_PROG = "nanollm-train"
LOGGER_NAME = "scripts.resume"

RunResumeForRunner = Callable[..., MagicMock]


@pytest.fixture
def run_resume(make_run_cli: MakeRunCli) -> RunCli:
    """Run the resume CLI and return everything it printed."""
    return make_run_cli(resume_main, RESUME_PROG)


@pytest.fixture
def run_train(make_run_cli: MakeRunCli) -> RunCli:
    """Run the train CLI, so a resume test can produce a checkpoint to resume from."""
    return make_run_cli(train_main, TRAIN_PROG)


class TestResumeCliHappyPath:
    def test_train_then_resume_doubles_cumulative_epochs(
        self,
        data_file: Path,
        checkpoint_path_factory: CheckpointPathFactory,
        run_train: RunCli,
        run_resume: RunCli,
    ) -> None:
        """End-to-end: train one epoch, then resume one more epoch. The
        resulting checkpoint's metadata must record cumulative_epochs_completed
        equal to the sum of both phases."""
        first_path = checkpoint_path_factory("resume_cli_test_first")
        second_path = checkpoint_path_factory("resume_cli_test_second")
        epochs_per_phase = 1

        run_train([
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", str(epochs_per_phase),
            "--batch-size", "2",
            "--checkpoint-destination", str(first_path),
        ])
        assert first_path.exists()

        run_resume([
            "--checkpoint-source", str(first_path),
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", str(epochs_per_phase),
            "--batch-size", "2",
            "--checkpoint-destination", str(second_path),
        ])

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
    def patched_run(self, data_file: Path, run_resume: RunCli) -> RunResumeForRunner:
        """Patches downstream training so only the source-resolution path is exercised."""

        def _run(*extra_args: str, latest: Path | None = None) -> MagicMock:
            args = [
                "--data-file", str(data_file),
                "--epochs", "1",
                "--batch-size", "2",
                *extra_args,
            ]
            with patch("src.cli.get_latest_checkpoint", return_value=latest), \
                 patch("scripts.resume.Runner") as mock_runner_cls:
                mock_runner_cls.return_value.run.return_value = None
                run_resume(args)
                return mock_runner_cls

        return _run

    def test_explicit_source_checkpoint_flag_forwarded(
        self, patched_run: RunResumeForRunner
    ) -> None:
        explicit_source = CHECKPOINTS_DIR / "explicit_source.orbax"
        mock_runner_cls = patched_run("--checkpoint-source", str(explicit_source))
        assert mock_runner_cls.call_args.kwargs["checkpoint_source"] == explicit_source

    def test_falls_back_to_latest_when_source_omitted(
        self, patched_run: RunResumeForRunner
    ) -> None:
        latest = CHECKPOINTS_DIR / "latest_auto.orbax"
        mock_runner_cls = patched_run(latest=latest)
        assert mock_runner_cls.call_args.kwargs["checkpoint_source"] == latest


class TestResumeCliErrors:
    CAPTURED_LOGGER = LOGGER_NAME

    def test_no_source_and_no_checkpoints_exits_1(
        self,
        data_file: Path,
        run_resume: RunCli,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When --checkpoint-source is omitted and no bundles exist, the CLI
        must exit 1 and log a clear error mentioning that no checkpoints were found."""
        with patch("src.cli.get_latest_checkpoint", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                run_resume([
                    "--data-file", str(data_file),
                    "--epochs", "1",
                    "--batch-size", "2",
                ])
        assert_error_exit(exc_info, caplog)
        assert "No checkpoints found" in caplog.text

    def test_nonexistent_source_checkpoint_exits_1(
        self,
        data_file: Path,
        run_resume: RunCli,
        missing_checkpoint_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If --checkpoint-source points at a missing bundle, the CLI must exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            run_resume([
                "--checkpoint-source", str(missing_checkpoint_path),
                "--data-file", str(data_file),
                "--epochs", "1",
                "--batch-size", "2",
            ])
        assert_error_exit(exc_info, caplog)
