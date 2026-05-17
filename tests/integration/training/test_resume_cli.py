"""Integration tests for the resume CLI entry point (scripts/resume.py)."""

import logging
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.resume import main as resume_main
from scripts.train import main as train_main
from src.checkpoint import load_metadata
from src.paths import CHECKPOINTS_DIR, DATA_DIR

# Enough stories for at least one batch with batch_size=2
_FAKE_STORIES = "\n".join(
    f"Once upon a time story number {i} ended here.<|endoftext|>" for i in range(6)
)


@pytest.fixture
def data_file() -> Generator[Path, None, None]:
    path = DATA_DIR / f"resume_cli_test_{uuid.uuid4().hex[:8]}.txt"
    path.write_text(_FAKE_STORIES, encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def checkpoint_path_factory() -> Generator[list[Path], None, None]:
    """Yields a list that collects checkpoint paths so the fixture can clean
    each one up at teardown without callers having to track them."""
    created: list[Path] = []

    def _new() -> Path:
        path = CHECKPOINTS_DIR / f"resume_cli_test_{uuid.uuid4().hex[:8]}.orbax"
        created.append(path)
        return path

    yield _new  # type: ignore[misc]
    for path in created:
        if path.exists():
            shutil.rmtree(path)


class TestResumeCliHappyPath:
    def test_train_then_resume_doubles_cumulative_epochs(
        self, data_file: Path, checkpoint_path_factory
    ) -> None:
        """End-to-end: train one epoch, then resume one more epoch. The
        resulting checkpoint's metadata must record cumulative_epochs_completed
        equal to the sum of both phases."""
        first_path = checkpoint_path_factory()
        second_path = checkpoint_path_factory()
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
    flow correctly into build_model_from_checkpoint. Trainer execution is
    patched so these tests stay fast and don't write real bundles."""

    @pytest.fixture
    def patched_run(self, data_file: Path):
        """Patches downstream training so only the source-resolution path is exercised."""

        def _run(argv: list[str], *, latest: Path | None = None) -> MagicMock:
            with patch("scripts.resume.build_model_from_checkpoint") as mock_build, \
                 patch("scripts.resume.ResumeContext.from_checkpoint") as mock_from_ckpt, \
                 patch("scripts.resume.get_latest_checkpoint", return_value=latest), \
                 patch("scripts.resume.count_params", return_value=0), \
                 patch("scripts.resume.Runner") as mock_runner_cls:
                mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
                mock_from_ckpt.return_value = MagicMock(
                    source=MagicMock(), previous_epochs_completed=3
                )
                mock_runner_cls.return_value.run.return_value = None
                with patch("sys.argv", argv):
                    resume_main()
                return mock_build

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
        mock_build = patched_run(argv)
        assert mock_build.call_args.args[0] == explicit_source

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
        mock_build = patched_run(argv, latest=latest)
        assert mock_build.call_args.args[0] == latest


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
