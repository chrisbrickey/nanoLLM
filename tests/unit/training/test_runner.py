"""Unit tests for src/training/runner.py

Tests exercise orchestration logic only — no disk access, no tokenization,
no JAX compilation.
"""

import dataclasses
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.training.resume_context import ResumeContext
from src.training.runner import run

SAMPLE_DATA_FILE = Path("/fake/data/stories.txt")
SAMPLE_CHECKPOINT_PATH = Path("/fake/checkpoints/run_01")
SAMPLE_CHECKPOINT_SOURCE = Path("/fake/checkpoints/run_00")
SAMPLE_PRIOR_EPOCHS = 5

SAMPLE_TRAINING_CONFIG = TrainingConfig()
SAMPLE_TOKENIZER_CONFIG = TokenizerConfig()
SAMPLE_MODEL_CONFIG = ModelConfig(
    maxlen=4,
    vocab_size=50,
    embed_dim=12,
    num_heads=3,
    feed_forward_dim=16,
    num_transformer_blocks=1,
)


def _make_mock_model() -> MagicMock:
    mock = MagicMock(spec=NanoLLM)
    mock.config = SAMPLE_MODEL_CONFIG
    return mock


def _default_run_kwargs(model: MagicMock | None = None) -> dict[str, object]:
    return dict(
        model=model or _make_mock_model(),
        tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        data_source=SAMPLE_DATA_FILE,
        training_config=SAMPLE_TRAINING_CONFIG,
        checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
    )


def _patch_pipeline(history: dict[str, list[float]] | None = None):
    """Returns a context manager that patches the data + training pipeline
    so runner tests can drive only the orchestration paths."""
    history = history if history is not None else {"train_loss": [0.7, 0.4]}

    class _Ctx:
        def __enter__(self) -> dict[str, MagicMock]:
            self.patches = {
                "load_text_from_file": patch("src.training.runner.load_text_from_file"),
                "calculate_batches": patch("src.training.runner.calculate_batches"),
                "preprocess_data": patch("src.training.runner.preprocess_data"),
                "Trainer": patch("src.training.runner.Trainer"),
                "save_checkpoint": patch("src.training.runner.save_checkpoint"),
            }
            entered = {name: ctx.__enter__() for name, ctx in self.patches.items()}
            entered["load_text_from_file"].return_value = ["story1", "story2"]
            entered["calculate_batches"].return_value = 4
            entered["preprocess_data"].return_value = MagicMock()
            trainer_instance = MagicMock()
            trainer_instance.train.return_value = history
            entered["Trainer"].return_value = trainer_instance
            self._entered = entered
            return entered

        def __exit__(self, exc_type, exc, tb) -> None:
            for ctx in self.patches.values():
                ctx.__exit__(exc_type, exc, tb)

    return _Ctx()


class TestRunDataPipeline:
    def test_raises_on_empty_dataset(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load:
            mock_load.return_value = []
            with pytest.raises(ValueError, match="Dataset is empty"):
                run(**_default_run_kwargs())

    def test_calls_load_text_with_correct_args(self) -> None:
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs())
            patched["load_text_from_file"].assert_called_once_with(
                file_path=SAMPLE_DATA_FILE,
                delimiter=SAMPLE_TOKENIZER_CONFIG.delimiter,
                max_paragraphs=SAMPLE_TRAINING_CONFIG.max_stories,
            )

    def test_calls_preprocess_with_correct_args(self) -> None:
        fake_stories = ["a", "b", "c"]
        mock_model = _make_mock_model()
        with _patch_pipeline() as patched:
            patched["load_text_from_file"].return_value = fake_stories
            run(**_default_run_kwargs(model=mock_model))
            patched["preprocess_data"].assert_called_once_with(
                text_blocks=fake_stories,
                model_config=mock_model.config,
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
                training_config=SAMPLE_TRAINING_CONFIG,
            )


class TestRunTrainerInvocation:
    def test_calls_trainer_train_once(self) -> None:
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs())
            patched["Trainer"].return_value.train.assert_called_once()

    def test_propagates_data_file_not_found(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load:
            mock_load.side_effect = FileNotFoundError("missing file")
            with pytest.raises(FileNotFoundError):
                run(**_default_run_kwargs())

    def test_propagates_data_os_error(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load:
            mock_load.side_effect = OSError("disk error")
            with pytest.raises(OSError):
                run(**_default_run_kwargs())

    def test_propagates_calculate_batches_error(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc:
            mock_load.return_value = ["story1"]
            mock_calc.side_effect = ValueError("bad batch size")
            with pytest.raises(ValueError):
                run(**_default_run_kwargs())

    def test_propagates_training_value_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = ValueError("training failed")
            with pytest.raises(ValueError):
                run(**_default_run_kwargs())

    def test_propagates_training_runtime_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = RuntimeError("runtime failure")
            with pytest.raises(RuntimeError):
                run(**_default_run_kwargs())

    def test_propagates_training_os_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = OSError("disk full")
            with pytest.raises(OSError):
                run(**_default_run_kwargs())


class TestRunCheckpointPersistence:
    def test_save_invoked_with_destination_path(self) -> None:
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs())
            patched["save_checkpoint"].assert_called_once()
            args, kwargs = patched["save_checkpoint"].call_args
            assert args[1] == SAMPLE_CHECKPOINT_PATH

    def test_save_not_invoked_when_destination_is_none(self) -> None:
        kwargs = _default_run_kwargs()
        kwargs["checkpoint_destination"] = None
        with _patch_pipeline() as patched:
            run(**kwargs)
            patched["save_checkpoint"].assert_not_called()

    def test_metadata_records_tokenizer_config(self) -> None:
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs())
            metadata = patched["save_checkpoint"].call_args.kwargs["metadata"]
            assert metadata.tokenizer_config == dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

    def test_metadata_records_final_loss_from_history(self) -> None:
        history = {"train_loss": [0.9, 0.6, 0.3]}
        with _patch_pipeline(history=history) as patched:
            run(**_default_run_kwargs())
            metadata = patched["save_checkpoint"].call_args.kwargs["metadata"]
            assert metadata.final_loss == 0.3

    def test_metadata_final_loss_is_none_for_empty_history(self) -> None:
        with _patch_pipeline(history={"train_loss": []}) as patched:
            run(**_default_run_kwargs())
            metadata = patched["save_checkpoint"].call_args.kwargs["metadata"]
            assert metadata.final_loss is None


class TestRunResumeContext:
    def test_resume_from_none_records_zero_prior_epochs(self) -> None:
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs())
            metadata = patched["save_checkpoint"].call_args.kwargs["metadata"]
            assert metadata.cumulative_epochs_completed == SAMPLE_TRAINING_CONFIG.epochs

    def test_resume_from_adds_prior_epochs_to_cumulative(self) -> None:
        ctx = ResumeContext(source=SAMPLE_CHECKPOINT_SOURCE, previous_epochs_completed=SAMPLE_PRIOR_EPOCHS)
        with _patch_pipeline() as patched:
            run(**_default_run_kwargs(), resume_from=ctx)
            metadata = patched["save_checkpoint"].call_args.kwargs["metadata"]
            assert metadata.cumulative_epochs_completed == (
                SAMPLE_PRIOR_EPOCHS + SAMPLE_TRAINING_CONFIG.epochs
            )


class TestRunHeaderLog:
    def test_header_logs_data_source_and_destination(self, caplog: pytest.LogCaptureFixture) -> None:
        with _patch_pipeline(), caplog.at_level(logging.INFO, logger="src.training.runner"):
            run(**_default_run_kwargs())
        assert str(SAMPLE_DATA_FILE) in caplog.text
        assert str(SAMPLE_CHECKPOINT_PATH) in caplog.text

    def test_header_logs_resume_lines_when_resuming(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = ResumeContext(source=SAMPLE_CHECKPOINT_SOURCE, previous_epochs_completed=SAMPLE_PRIOR_EPOCHS)
        with _patch_pipeline(), caplog.at_level(logging.INFO, logger="src.training.runner"):
            run(**_default_run_kwargs(), resume_from=ctx)
        assert str(SAMPLE_CHECKPOINT_SOURCE) in caplog.text
        assert f"previous epochs trained: {SAMPLE_PRIOR_EPOCHS}" in caplog.text

    def test_cumulative_epoch_summary_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = ResumeContext(source=SAMPLE_CHECKPOINT_SOURCE, previous_epochs_completed=SAMPLE_PRIOR_EPOCHS)
        with _patch_pipeline(), caplog.at_level(logging.INFO, logger="src.training.runner"):
            run(**_default_run_kwargs(), resume_from=ctx)
        assert f"in addition to {SAMPLE_PRIOR_EPOCHS} epochs accumulated" in caplog.text
        assert f"Cumulative epochs completed: {SAMPLE_PRIOR_EPOCHS + SAMPLE_TRAINING_CONFIG.epochs}" in caplog.text

    def test_no_destination_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        kwargs = _default_run_kwargs()
        kwargs["checkpoint_destination"] = None
        with _patch_pipeline(), caplog.at_level(logging.WARNING, logger="src.training.runner"):
            run(**kwargs)
        assert "no checkpoint will be persisted" in caplog.text
