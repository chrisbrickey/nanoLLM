"""Unit tests for src/utils.py"""

from pathlib import Path
from unittest.mock import mock_open, patch
import pytest

from src.utils import load_stories_from_file

TEST_FILE_PATH = "/path/to/test_file.txt"
DELIMITER = "<|endoftext|>"
SAMPLE_STORY_1 = "Once upon a time in a test."
SAMPLE_STORY_2 = "Another sample story for testing."
SAMPLE_STORY_3 = "Third test story goes here."


class TestLoadStoriesFromFile:
    """Test suite for load_stories_from_file utility function"""

    def test_load_single_story_with_delimiter(self, capsys):
        """Test loading a single story that ends with delimiter"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_stories_from_file(TEST_FILE_PATH)

        assert len(stories) == 1
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"

        # Verify print output
        captured = capsys.readouterr()
        assert "Loading stories" in captured.out
        assert "Loaded 1 stories" in captured.out

    def test_load_multiple_stories(self, capsys):
        """Test loading multiple stories separated by delimiters"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_stories_from_file(TEST_FILE_PATH)

        assert len(stories) == 3
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"
        assert stories[2] == f"{SAMPLE_STORY_3}{DELIMITER}"

        captured = capsys.readouterr()
        assert "Loaded 3 stories" in captured.out

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist"""
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                load_stories_from_file(TEST_FILE_PATH)

    def test_empty_file(self, capsys):
        """Test loading from an empty file returns empty list"""
        file_content = ""

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_stories_from_file(TEST_FILE_PATH)

        assert stories == []

        captured = capsys.readouterr()
        assert "Loaded 0 stories" in captured.out

    def test_max_stories_limit(self, capsys):
        """Test that max_stories parameter limits the number of stories loaded"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_stories_from_file(TEST_FILE_PATH, max_stories=2)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"

        captured = capsys.readouterr()
        assert "Loaded 2 stories" in captured.out

    def test_multiple_delimiters_on_same_line(self, capsys):
        """Test handling multiple delimiters on a single line"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}{SAMPLE_STORY_2}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_stories_from_file(TEST_FILE_PATH)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"
