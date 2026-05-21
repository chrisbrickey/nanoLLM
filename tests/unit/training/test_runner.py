"""Unit tests for src/training/runner.py

Tests exercise orchestration logic only — no disk access, no tokenization,
no JAX compilation.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.training.runner import Runner
from src.training.schema import CheckpointMetadata, MetricsHistory

SAMPLE_DATA_FILE = Path("/fake/data/stories.txt")
SAMPLE_CHECKPOINT_PATH = Path("/fake/checkpoints/run_01")
SAMPLE_CHECKPOINT_SOURCE = Path("/fake/checkpoints/prior_run")
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

# Stories provided to the load mock must satisfy
# len(stories) // training_config.batch_size > 0 so that Runner's
# internal batches-per-epoch check passes.
_STORIES_PER_BATCH_MULTIPLIER = 4
SAMPLE_STORIES = [
    f"story{i}"
    for i in range(SAMPLE_TRAINING_CONFIG.batch_size * _STORIES_PER_BATCH_MULTIPLIER)
]


def _make_mock_model() -> MagicMock:
    mock = MagicMock(spec=NanoLLM)
    mock.config = SAMPLE_MODEL_CONFIG
    return mock


def _default_runner_kwargs() -> dict[str, object]:
    """Default kwargs for the fresh-training pathway."""
    return dict(
        model_config=SAMPLE_MODEL_CONFIG,
        tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        data_source=SAMPLE_DATA_FILE,
        training_config=SAMPLE_TRAINING_CONFIG,
        checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
    )


def _run_default(**overrides: object) -> MetricsHistory:
    """Construct a Runner with default kwargs (optionally overridden) and call .run()."""
    kwargs = _default_runner_kwargs()
    kwargs.update(overrides)
    # Filter out keys whose values are None when caller wants to drop them
    # (e.g. switching from fresh-training to resume requires removing model_config).
    kwargs = {k: v for k, v in kwargs.items() if v is not None or k == "checkpoint_destination"}
    return Runner(**kwargs).run()


def _patch_pipeline(
    history: MetricsHistory | None = None,
    mock_model: MagicMock | None = None,
    restored_tuple: tuple | None = None,
):
    """Returns a context manager that patches the data + training pipeline
    so runner tests can drive only the orchestration paths.

    Always patches NanoLLM and restore_from_checkpoint so neither pathway
    constructs/restores real weights. Callers may inject a fake
    restore_from_checkpoint return value via restored_tuple.
    """
    history = history if history is not None else MetricsHistory(train_loss=[0.7, 0.4])
    mock_model = mock_model or _make_mock_model()

    class _Ctx:
        def __enter__(self) -> dict[str, MagicMock]:
            self.patches = {
                "load_text_from_file": patch("src.training.runner.load_text_from_file"),
                "Processor": patch("src.training.runner.Processor"),
                "Trainer": patch("src.training.runner.Trainer"),
                "build_and_save_checkpoint": patch("src.training.runner.build_and_save_checkpoint"),
                "NanoLLM": patch("src.training.runner.NanoLLM", return_value=mock_model),
                "count_params": patch("src.training.runner.count_params", return_value=42),
                "restore_from_checkpoint": patch("src.training.runner.restore_from_checkpoint"),
            }
            entered = {name: ctx.__enter__() for name, ctx in self.patches.items()}
            entered["load_text_from_file"].return_value = list(SAMPLE_STORIES)
            processor_instance = MagicMock()
            processor_instance.process.return_value = MagicMock()
            entered["Processor"].return_value = processor_instance
            trainer_instance = MagicMock()
            trainer_instance.train.return_value = history
            entered["Trainer"].return_value = trainer_instance
            if restored_tuple is not None:
                entered["restore_from_checkpoint"].return_value = restored_tuple
            self._entered = entered
            return entered

        def __exit__(self, exc_type, exc, tb) -> None:
            for ctx in self.patches.values():
                ctx.__exit__(exc_type, exc, tb)

    return _Ctx()


def _resume_overrides() -> dict[str, object]:
    """Helper to convert default fresh-training kwargs into resume kwargs."""
    return dict(
        model_config=None,
        tokenizer_config=None,
        checkpoint_source=SAMPLE_CHECKPOINT_SOURCE,
    )


def _restored(prior_epochs: int = SAMPLE_PRIOR_EPOCHS) -> tuple:
    """Standard restore_from_checkpoint return value used by resume-path tests."""
    return (
        _make_mock_model(),
        SAMPLE_TOKENIZER_CONFIG,
        CheckpointMetadata(cumulative_epochs_completed=prior_epochs),
    )


class TestConstructorValidation:
    def test_raises_when_no_model_inputs_provided(self) -> None:
        with pytest.raises(ValueError, match="model_config"):
            Runner(
                data_source=SAMPLE_DATA_FILE,
                training_config=SAMPLE_TRAINING_CONFIG,
                checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
            )

    def test_raises_when_only_model_config_provided(self) -> None:
        with pytest.raises(ValueError, match="tokenizer_config"):
            Runner(
                data_source=SAMPLE_DATA_FILE,
                training_config=SAMPLE_TRAINING_CONFIG,
                checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
                model_config=SAMPLE_MODEL_CONFIG,
            )

    def test_raises_when_only_tokenizer_config_provided(self) -> None:
        with pytest.raises(ValueError, match="model_config"):
            Runner(
                data_source=SAMPLE_DATA_FILE,
                training_config=SAMPLE_TRAINING_CONFIG,
                checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
            )

    def test_accepts_checkpoint_source_alone(self) -> None:
        Runner(
            data_source=SAMPLE_DATA_FILE,
            training_config=SAMPLE_TRAINING_CONFIG,
            checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
            checkpoint_source=SAMPLE_CHECKPOINT_SOURCE,
        )

    def test_accepts_model_and_tokenizer_configs(self) -> None:
        Runner(
            data_source=SAMPLE_DATA_FILE,
            training_config=SAMPLE_TRAINING_CONFIG,
            checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
            model_config=SAMPLE_MODEL_CONFIG,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        )


class TestRunModelPreparation:
    def test_constructs_fresh_model_when_no_checkpoint_source(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            patched["NanoLLM"].assert_called_once_with(SAMPLE_MODEL_CONFIG)
            patched["restore_from_checkpoint"].assert_not_called()

    def test_restores_from_checkpoint_when_source_provided(self) -> None:
        with _patch_pipeline(restored_tuple=_restored()) as patched:
            _run_default(**_resume_overrides())
            patched["restore_from_checkpoint"].assert_called_once_with(SAMPLE_CHECKPOINT_SOURCE)
            patched["NanoLLM"].assert_not_called()

    def test_supersede_warning_when_both_configs_and_checkpoint_source_passed(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with _patch_pipeline(restored_tuple=_restored()), caplog.at_level(
            logging.WARNING, logger="src.training.runner"
        ):
            _run_default(checkpoint_source=SAMPLE_CHECKPOINT_SOURCE)
        assert any(
            "checkpoint_source" in r.message
            and "ignored" in r.message
            and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_no_supersede_warning_when_only_checkpoint_source_passed(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with _patch_pipeline(restored_tuple=_restored(prior_epochs=0)), caplog.at_level(
            logging.WARNING, logger="src.training.runner"
        ):
            _run_default(**_resume_overrides())
        assert not any(
            "ignored in favor of values from the checkpoint" in r.message
            for r in caplog.records
        )


class TestRunDataPipeline:
    def test_raises_on_empty_dataset(self) -> None:
        with _patch_pipeline() as patched:
            patched["load_text_from_file"].return_value = []
            with pytest.raises(ValueError, match="Dataset is empty"):
                _run_default()

    def test_calls_load_text_with_correct_args(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            patched["load_text_from_file"].assert_called_once_with(
                file_path=SAMPLE_DATA_FILE,
                delimiter=SAMPLE_TOKENIZER_CONFIG.delimiter,
                max_paragraphs=SAMPLE_TRAINING_CONFIG.max_stories,
            )

    def test_calls_processor_with_correct_args(self) -> None:
        fake_stories = list(SAMPLE_STORIES)
        mock_model = _make_mock_model()
        with _patch_pipeline(mock_model=mock_model) as patched:
            patched["load_text_from_file"].return_value = fake_stories
            _run_default()
            patched["Processor"].assert_called_once_with(
                model_config=mock_model.config,
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
                training_config=SAMPLE_TRAINING_CONFIG,
            )
            patched["Processor"].return_value.process.assert_called_once_with(fake_stories)

    def test_raises_when_batch_size_exceeds_record_count(self) -> None:
        """calculate_batches is now an internal method; verify it still
        aborts when record_count // batch_size <= 0."""
        oversized_batch = TrainingConfig(batch_size=10)
        with _patch_pipeline() as patched:
            patched["load_text_from_file"].return_value = ["only_one_story"]
            with pytest.raises(ValueError, match="batches per epoch"):
                _run_default(training_config=oversized_batch)


class TestRunTrainerInvocation:
    def test_calls_trainer_train_once(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            patched["Trainer"].return_value.train.assert_called_once()

    def test_returns_metrics_history_from_trainer(self) -> None:
        history = MetricsHistory(train_loss=[0.8, 0.5, 0.2])
        with _patch_pipeline(history=history):
            result = _run_default()
        assert result == history

    def test_propagates_data_file_not_found(self) -> None:
        with _patch_pipeline() as patched:
            patched["load_text_from_file"].side_effect = FileNotFoundError("missing file")
            with pytest.raises(FileNotFoundError):
                _run_default()

    def test_propagates_data_os_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["load_text_from_file"].side_effect = OSError("disk error")
            with pytest.raises(OSError):
                _run_default()

    def test_propagates_training_value_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = ValueError("training failed")
            with pytest.raises(ValueError):
                _run_default()

    def test_propagates_training_runtime_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = RuntimeError("runtime failure")
            with pytest.raises(RuntimeError):
                _run_default()

    def test_propagates_training_os_error(self) -> None:
        with _patch_pipeline() as patched:
            patched["Trainer"].return_value.train.side_effect = OSError("disk full")
            with pytest.raises(OSError):
                _run_default()


class TestRunCheckpointPersistence:
    def test_save_invoked_with_destination_path(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            patched["build_and_save_checkpoint"].assert_called_once()
            args, _ = patched["build_and_save_checkpoint"].call_args
            assert args[1] == SAMPLE_CHECKPOINT_PATH

    def test_save_not_invoked_when_destination_is_none(self) -> None:
        with _patch_pipeline() as patched:
            _run_default(checkpoint_destination=None)
            patched["build_and_save_checkpoint"].assert_not_called()

    def test_forwards_tokenizer_config_to_persistence(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            call = patched["build_and_save_checkpoint"].call_args
            assert call.kwargs["tokenizer_config"] == SAMPLE_TOKENIZER_CONFIG

    def test_forwards_final_loss_to_persistence(self) -> None:
        history = MetricsHistory(train_loss=[0.9, 0.6, 0.3])
        with _patch_pipeline(history=history) as patched:
            _run_default()
            call = patched["build_and_save_checkpoint"].call_args
            assert call.kwargs["final_loss"] == 0.3

    def test_forwards_none_final_loss_for_empty_history(self) -> None:
        with _patch_pipeline(history=MetricsHistory()) as patched:
            _run_default()
            call = patched["build_and_save_checkpoint"].call_args
            assert call.kwargs["final_loss"] is None


class TestRunCumulativeEpochs:
    def test_zero_prior_epochs_when_no_checkpoint_source(self) -> None:
        with _patch_pipeline() as patched:
            _run_default()
            call = patched["build_and_save_checkpoint"].call_args
            assert call.kwargs["cumulative_epochs_completed"] == SAMPLE_TRAINING_CONFIG.epochs

    def test_prior_epochs_loaded_from_checkpoint_source(self) -> None:
        with _patch_pipeline(restored_tuple=_restored()) as patched:
            _run_default(**_resume_overrides())
            call = patched["build_and_save_checkpoint"].call_args
            assert call.kwargs["cumulative_epochs_completed"] == (
                SAMPLE_PRIOR_EPOCHS + SAMPLE_TRAINING_CONFIG.epochs
            )


class TestRunHeaderLog:
    def test_header_logs_data_source_and_destination(self, caplog: pytest.LogCaptureFixture) -> None:
        with _patch_pipeline(), caplog.at_level(logging.INFO, logger="src.training.runner"):
            _run_default()
        assert str(SAMPLE_DATA_FILE) in caplog.text
        assert str(SAMPLE_CHECKPOINT_PATH) in caplog.text

    def test_header_logs_previous_epochs_when_resuming(self, caplog: pytest.LogCaptureFixture) -> None:
        with _patch_pipeline(restored_tuple=_restored()), caplog.at_level(
            logging.INFO, logger="src.training.runner"
        ):
            _run_default(**_resume_overrides())
        assert f"previous epochs trained: {SAMPLE_PRIOR_EPOCHS}" in caplog.text

    def test_cumulative_epoch_summary_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with _patch_pipeline(restored_tuple=_restored()), caplog.at_level(
            logging.INFO, logger="src.training.runner"
        ):
            _run_default(**_resume_overrides())
        assert f"in addition to {SAMPLE_PRIOR_EPOCHS} epochs accumulated" in caplog.text
        assert f"Cumulative epochs completed: {SAMPLE_PRIOR_EPOCHS + SAMPLE_TRAINING_CONFIG.epochs}" in caplog.text

    def test_no_destination_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with _patch_pipeline(), caplog.at_level(logging.WARNING, logger="src.training.runner"):
            _run_default(checkpoint_destination=None)
        assert "no checkpoint will be persisted" in caplog.text
