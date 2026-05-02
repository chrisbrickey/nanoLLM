"""Unit tests for StoryDataset — tokenizer is mocked."""

from unittest.mock import MagicMock, patch
import pytest

from src.data.dataset import StoryDataset

SAMPLE_STORY_001 = "sample story 001"
SAMPLE_STORY_002 = "sample story 002 with additional content"
TEST_DELIMITER = "<|delimiter|>"
TEST_MAXLEN = 10
DELIMITER_TOKEN = 999


def _make_mock_tokenizer() -> MagicMock:
    """Return a tokenizer mock where encode maps text → range(len(text)) tokens."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))
    return tok


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    return _make_mock_tokenizer()


@pytest.fixture
def dataset(mock_tokenizer: MagicMock) -> StoryDataset:
    with patch("src.data.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
        mock_tokenizer.encode.return_value = [DELIMITER_TOKEN]
        ds = StoryDataset(
            stories=[SAMPLE_STORY_001, SAMPLE_STORY_002],
            maxlen=TEST_MAXLEN,
            delimiter=TEST_DELIMITER,
        )
    ds.delimiter_token = DELIMITER_TOKEN
    mock_tokenizer.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))
    return ds


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
        with patch("src.data.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
            mock_tokenizer.encode.return_value = [DELIMITER_TOKEN]
            ds = StoryDataset(stories=[long_story], maxlen=TEST_MAXLEN, delimiter=TEST_DELIMITER)
        ds.delimiter_token = DELIMITER_TOKEN
        mock_tokenizer.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))

        result = ds[0]

        assert len(result) == TEST_MAXLEN
        assert result == list(range(TEST_MAXLEN))

    def test_pads_short_sequence_with_zeros(self, mock_tokenizer: MagicMock) -> None:
        short_story = "abc"  # 3 tokens with mock
        with patch("src.data.dataset.tiktoken.get_encoding", return_value=mock_tokenizer):
            mock_tokenizer.encode.return_value = [DELIMITER_TOKEN]
            ds = StoryDataset(stories=[short_story], maxlen=TEST_MAXLEN, delimiter=TEST_DELIMITER)
        ds.delimiter_token = DELIMITER_TOKEN
        mock_tokenizer.encode.side_effect = lambda text, allowed_special=None: list(range(len(text)))

        result = ds[0]

        assert result[:3] == [0, 1, 2]
        assert result[3:] == [0] * (TEST_MAXLEN - 3)
