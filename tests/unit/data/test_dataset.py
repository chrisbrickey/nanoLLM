"""Unit tests for StoryDataset — tokenizer is mocked."""

from unittest.mock import MagicMock, PropertyMock, patch
import pytest

from src.config import TokenizerConfig
from src.data.dataset import StoryDataset

SAMPLE_STORY_001 = "sample story 001"
SAMPLE_STORY_002 = "sample story 002 with additional content"
TEST_DELIMITER = "<|delimiter|>"
TEST_MAXLEN = 10
DELIMITER_TOKEN = 999
PAD_TOKEN_ID = 0


def _make_mock_tokenizer() -> MagicMock:
    """Return a tokenizer mock where encode maps text → range(len(text)) tokens."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))
    return tok


def _make_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        delimiter=TEST_DELIMITER, name="gpt2", pad_token_id=PAD_TOKEN_ID
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    return _make_mock_tokenizer()


def _build_dataset(
    stories: list[str], mock_tok: MagicMock
) -> StoryDataset:
    """Build a StoryDataset whose TokenizerConfig.tokenizer property returns mock_tok."""
    with patch.object(
        TokenizerConfig, "tokenizer", new_callable=PropertyMock
    ) as mock_property:
        mock_property.return_value = mock_tok
        # During __init__, encode() resolves the delimiter token first.
        mock_tok.encode.return_value = [DELIMITER_TOKEN]
        ds = StoryDataset(
            stories=stories, maxlen=TEST_MAXLEN, tokenizer_config=_make_tokenizer_config()
        )
    # After construction, switch to length-based encoding for __getitem__ tests.
    mock_tok.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))
    return ds


@pytest.fixture
def dataset(mock_tokenizer: MagicMock) -> StoryDataset:
    return _build_dataset([SAMPLE_STORY_001, SAMPLE_STORY_002], mock_tokenizer)


class TestStoryDatasetLen:
    def test_returns_number_of_stories(self, dataset: StoryDataset) -> None:
        assert len(dataset) == 2


class TestStoryDatasetGetitem:
    def test_returns_list_of_ints(self, dataset: StoryDataset) -> None:
        result = dataset[0]
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_output_length_equals_maxlen(self, dataset: StoryDataset) -> None:
        assert len(dataset[0]) == TEST_MAXLEN

    def test_truncates_long_sequence(self, mock_tokenizer: MagicMock) -> None:
        long_story = "x" * 50  # 50 tokens with mock
        ds = _build_dataset([long_story], mock_tokenizer)

        result = ds[0]

        assert len(result) == TEST_MAXLEN
        assert result == list(range(TEST_MAXLEN))

    def test_pads_short_sequence_with_pad_token(self, mock_tokenizer: MagicMock) -> None:
        short_story = "abc"  # 3 tokens with mock
        ds = _build_dataset([short_story], mock_tokenizer)

        result = ds[0]

        assert result[:3] == [0, 1, 2]
        assert result[3:] == [PAD_TOKEN_ID] * (TEST_MAXLEN - 3)
