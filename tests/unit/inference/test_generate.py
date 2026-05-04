"""Unit tests for generate_text — model and tokenizer are mocked."""

import logging
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.inference.generate import generate_text

VOCAB_SIZE = 100
MAXLEN = 10
START_TOKENS = [1, 2, 3]
NEXT_TOKEN = 42
END_TOKEN_ID = 99
DELIMITER = "<|end|>"


def _logits_for_token(token_id: int) -> np.ndarray:
    """Return logits array where token_id has the highest score."""
    logits = np.zeros((1, MAXLEN, VOCAB_SIZE))
    logits[0, :, token_id] = 10.0
    return logits


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.maxlen = MAXLEN
    model.return_value = _logits_for_token(NEXT_TOKEN)
    return model


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.encode.return_value = [END_TOKEN_ID]
    tok.decode.return_value = "decoded text"
    return tok


class TestGenerateTextStopping:
    def test_stops_at_end_token(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_model.return_value = _logits_for_token(END_TOKEN_ID)

        with caplog.at_level(logging.INFO, logger="src.inference.generate"):
            generate_text(
                model=mock_model,
                tokenizer=mock_tokenizer,
                delimiter=DELIMITER,
                start_tokens=START_TOKENS,
                max_new_tokens=10,
            )

        assert mock_model.call_count == 1
        assert "stopped at end token" in caplog.text

    def test_generates_up_to_max_new_tokens(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        max_new = 5
        with caplog.at_level(logging.INFO, logger="src.inference.generate"):
            generate_text(
                model=mock_model,
                tokenizer=mock_tokenizer,
                delimiter=DELIMITER,
                start_tokens=START_TOKENS,
                max_new_tokens=max_new,
            )

        assert mock_model.call_count == max_new
        assert "Generation complete" in caplog.text
        assert "stopped at end token" not in caplog.text


class TestGenerateTextOutput:
    def test_returns_decoded_string(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="src.inference.generate"):
            result = generate_text(
                model=mock_model,
                tokenizer=mock_tokenizer,
                delimiter=DELIMITER,
                start_tokens=START_TOKENS,
                max_new_tokens=3,
            )

        assert isinstance(result, str)
        mock_tokenizer.decode.assert_called_once()
        assert "Generating text" in caplog.text

    def test_decode_receives_start_tokens_plus_generated(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        max_new = 3
        generate_text(
            model=mock_model,
            tokenizer=mock_tokenizer,
            delimiter=DELIMITER,
            start_tokens=START_TOKENS,
            max_new_tokens=max_new,
        )

        decoded_args = mock_tokenizer.decode.call_args[0][0]
        assert decoded_args[: len(START_TOKENS)] == START_TOKENS
        assert len(decoded_args) == len(START_TOKENS) + max_new


class TestGenerateTextPadding:
    def test_pads_short_context_to_maxlen(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        generate_text(
            model=mock_model,
            tokenizer=mock_tokenizer,
            delimiter=DELIMITER,
            start_tokens=START_TOKENS,  # len 3 < MAXLEN 10
            max_new_tokens=1,
        )

        call_args = mock_model.call_args[0][0]
        assert call_args.shape == (1, MAXLEN)
