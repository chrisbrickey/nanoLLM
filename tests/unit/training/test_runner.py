"""Unit tests for src/training/runner.py

Tests exercise orchestration logic only — no disk access, no tokenization,
no JAX compilation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.training.runner import run

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


class TestRun:
    """Test suite for run()"""

    def test_raises_on_empty_dataset(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load:
            mock_load.return_value = []
            with pytest.raises(ValueError, match="Dataset is empty"):
                run(**_default_run_kwargs())

    def test_calls_load_text_with_correct_args(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1", "story2"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_trainer_cls.return_value = MagicMock()

            run(**_default_run_kwargs())

            mock_load.assert_called_once_with(
                file_path=SAMPLE_DATA_FILE,
                delimiter=SAMPLE_TOKENIZER_CONFIG.delimiter,
                max_paragraphs=SAMPLE_TRAINING_CONFIG.max_stories,
            )

    def test_calls_preprocess_with_correct_args(self) -> None:
        fake_stories = ["a", "b", "c"]
        mock_model = _make_mock_model()
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = fake_stories
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_trainer_cls.return_value = MagicMock()

            run(**_default_run_kwargs(model=mock_model))

            mock_preprocess.assert_called_once_with(
                text_blocks=fake_stories,
                model_config=mock_model.config,
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
                training_config=SAMPLE_TRAINING_CONFIG,
            )

    def test_calls_trainer_train_once(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_trainer_cls.return_value = mock_instance

            run(**_default_run_kwargs())

            mock_instance.train.assert_called_once()

    def test_passes_checkpoint_source_to_trainer(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_trainer_cls.return_value = MagicMock()

            run(**_default_run_kwargs(), checkpoint_source=SAMPLE_CHECKPOINT_SOURCE)

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["checkpoint_source"] == SAMPLE_CHECKPOINT_SOURCE

    def test_uses_default_none_checkpoint_source(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_trainer_cls.return_value = MagicMock()

            run(**_default_run_kwargs())

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["checkpoint_source"] is None

    def test_passes_previous_epochs_completed_to_trainer(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_trainer_cls.return_value = MagicMock()

            run(**_default_run_kwargs(), previous_epochs_completed=5)

            _, kwargs = mock_trainer_cls.call_args
            assert kwargs["previous_epochs_completed"] == 5

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
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_instance.train.side_effect = ValueError("training failed")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(ValueError):
                run(**_default_run_kwargs())

    def test_propagates_training_runtime_error(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_instance.train.side_effect = RuntimeError("runtime failure")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(RuntimeError):
                run(**_default_run_kwargs())

    def test_propagates_training_os_error(self) -> None:
        with patch("src.training.runner.load_text_from_file") as mock_load, \
             patch("src.training.runner.calculate_batches") as mock_calc, \
             patch("src.training.runner.preprocess_data") as mock_preprocess, \
             patch("src.training.runner.Trainer") as mock_trainer_cls:
            mock_load.return_value = ["story1"]
            mock_calc.return_value = 4
            mock_preprocess.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_instance.train.side_effect = OSError("disk full")
            mock_trainer_cls.return_value = mock_instance

            with pytest.raises(OSError):
                run(**_default_run_kwargs())
