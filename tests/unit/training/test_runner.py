"""Unit tests for src/training/runner.py

Both public functions are tested by mocking their I/O dependencies so that
tests exercise orchestration logic only — no disk access, no tokenization,
no JAX compilation.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.training.runner import execute_training_run, prepare_dataloader

SAMPLE_DATA_FILE = Path("/fake/data/stories.txt")
SAMPLE_CHECKPOINT_PATH = Path("/fake/checkpoints/run_01")
SAMPLE_CHECKPOINT_SOURCE = Path("/fake/checkpoints/run_00")

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
    return MagicMock(spec=NanoLLM)


def _default_run_kwargs() -> dict[str, object]:
    return dict(
        model=_make_mock_model(),
        model_config=SAMPLE_MODEL_CONFIG,
        tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        data_source=SAMPLE_DATA_FILE,
        training_config=SAMPLE_TRAINING_CONFIG,
        checkpoint_destination=SAMPLE_CHECKPOINT_PATH,
    )


class TestPrepareDataloader:
    """Test suite for prepare_dataloader()"""

    def test_returns_dataloader_and_batches_per_epoch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_dl = MagicMock()
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.preprocess_data") as mock_preprocess:
            mock_load.return_value = ["story1", "story2"]
            mock_preprocess.return_value = (fake_dl, 7)

            with caplog.at_level(logging.INFO, logger="src.training.runner"):
                result = prepare_dataloader(
                    SAMPLE_MODEL_CONFIG, SAMPLE_TOKENIZER_CONFIG,
                    SAMPLE_DATA_FILE, SAMPLE_TRAINING_CONFIG,
                )

            assert result == (fake_dl, 7)
            assert "Loading data" in caplog.text
            assert "Data processing complete" in caplog.text

    def test_calls_load_text_with_correct_args(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.preprocess_data") as mock_preprocess:
            mock_load.return_value = []
            mock_preprocess.return_value = (MagicMock(), 0)

            prepare_dataloader(
                SAMPLE_MODEL_CONFIG, SAMPLE_TOKENIZER_CONFIG,
                SAMPLE_DATA_FILE, SAMPLE_TRAINING_CONFIG,
            )

            mock_load.assert_called_once_with(
                SAMPLE_DATA_FILE,
                delimiter=SAMPLE_TOKENIZER_CONFIG.delimiter,
                max_paragraphs=SAMPLE_TRAINING_CONFIG.max_stories,
            )

    def test_calls_preprocess_with_correct_args(self) -> None:
        fake_stories = ["a", "b", "c"]
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.preprocess_data") as mock_preprocess:
            mock_load.return_value = fake_stories
            mock_preprocess.return_value = (MagicMock(), 3)

            prepare_dataloader(
                SAMPLE_MODEL_CONFIG, SAMPLE_TOKENIZER_CONFIG,
                SAMPLE_DATA_FILE, SAMPLE_TRAINING_CONFIG,
            )

            mock_preprocess.assert_called_once_with(
                fake_stories,
                batch_size=SAMPLE_TRAINING_CONFIG.batch_size,
                maxlen=SAMPLE_MODEL_CONFIG.maxlen,
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
                shuffle=SAMPLE_TRAINING_CONFIG.shuffle,
                seed=SAMPLE_TRAINING_CONFIG.seed,
            )


class TestExecuteTrainingRun:
    """Test suite for execute_training_run()"""

    def test_calls_trainer_train_once(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_instance = MagicMock()
            mock_trainer_cls.return_value = mock_instance

            execute_training_run(**_default_run_kwargs())

            mock_instance.train.assert_called_once()

    def test_passes_checkpoint_source_to_trainer(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_trainer_cls.return_value = MagicMock()

            execute_training_run(
                **_default_run_kwargs(), checkpoint_source=SAMPLE_CHECKPOINT_SOURCE
            )

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["checkpoint_source"] == SAMPLE_CHECKPOINT_SOURCE

    def test_uses_default_none_checkpoint_source(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_trainer_cls.return_value = MagicMock()

            execute_training_run(**_default_run_kwargs())

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["checkpoint_source"] is None

    def test_passes_previous_epochs_completed_to_trainer(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_trainer_cls.return_value = MagicMock()

            execute_training_run(**_default_run_kwargs(), previous_epochs_completed=5)

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["previous_epochs_completed"] == 5

    def test_propagates_data_file_not_found(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer"):
            mock_prepare.side_effect = FileNotFoundError("missing file")

            with pytest.raises(FileNotFoundError):
                execute_training_run(**_default_run_kwargs())

    def test_propagates_data_value_error(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer"):
            mock_prepare.side_effect = ValueError("bad data")

            with pytest.raises(ValueError):
                execute_training_run(**_default_run_kwargs())

    def test_propagates_data_os_error(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer"):
            mock_prepare.side_effect = OSError("disk error")

            with pytest.raises(OSError):
                execute_training_run(**_default_run_kwargs())

    def test_propagates_training_value_error(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_instance = MagicMock()
            mock_instance.train.side_effect = ValueError("training failed")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(ValueError):
                execute_training_run(**_default_run_kwargs())

    def test_propagates_training_runtime_error(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_instance = MagicMock()
            mock_instance.train.side_effect = RuntimeError("runtime failure")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(RuntimeError):
                execute_training_run(**_default_run_kwargs())

    def test_propagates_training_os_error(self) -> None:
        with patch("src.training.runner.prepare_dataloader") as mock_prepare, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_prepare.return_value = (MagicMock(), 4)
            mock_instance = MagicMock()
            mock_instance.train.side_effect = OSError("disk full")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(OSError):
                execute_training_run(**_default_run_kwargs())
