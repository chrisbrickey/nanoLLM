"""Unit tests for src/data/loader.py"""

import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import mock_open, patch
import pytest

from src.data.loader import load_text_from_file
from src.config import PROJECT_ROOT

TEST_FILE_PATH = "data/test_file.txt"  # Relative to project root
DELIMITER = "<|endoftext|>"
SAMPLE_STORY_1 = "Once upon a time in a test."
SAMPLE_STORY_2 = "Another sample story for testing."
SAMPLE_STORY_3 = "Third test story goes here."


class TestLoadStoriesFromFile:
    """Test suite for load_stories_from_file utility function"""

    @pytest.fixture(autouse=True)
    def _capture_logs(self, caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
        with caplog.at_level(logging.INFO, logger="src.data.loader"):
            yield

    def test_load_single_story_with_delimiter(self, caplog):
        """Test loading a single story that ends with delimiter"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 1
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert "Loading data" in caplog.text
        assert "Loaded 1 paragraphs" in caplog.text

    def test_load_multiple_stories(self, caplog):
        """Test loading multiple stories separated by delimiters"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 3
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"
        assert stories[2] == f"{SAMPLE_STORY_3}{DELIMITER}"
        assert "Loaded 3 paragraphs" in caplog.text

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist"""
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                load_text_from_file(TEST_FILE_PATH, DELIMITER)

    def test_empty_file(self, caplog):
        """Test loading from an empty file returns empty list"""
        file_content = ""

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert stories == []
        assert "Loaded 0 paragraphs" in caplog.text

    def test_max_stories_limit(self, caplog):
        """Test that max_paragraphs parameter limits the number of paragraphs loaded"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n{SAMPLE_STORY_2}{DELIMITER}\n{SAMPLE_STORY_3}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER, max_paragraphs=2)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"
        assert "Loaded 2 paragraphs" in caplog.text

    def test_multiple_delimiters_on_same_line(self):
        """Test handling multiple delimiters on a single line"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}{SAMPLE_STORY_2}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                stories = load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert len(stories) == 2
        assert stories[0] == f"{SAMPLE_STORY_1}{DELIMITER}"
        assert stories[1] == f"{SAMPLE_STORY_2}{DELIMITER}"

    def test_path_outside_project_raises_error(self):
        """Test that paths outside project root are rejected"""
        with pytest.raises(ValueError, match="outside the project root"):
            load_text_from_file("../../etc/passwd", DELIMITER)

    def test_absolute_path_outside_project_raises_error(self):
        """Test that absolute paths outside project are rejected"""
        with pytest.raises(ValueError, match="outside the project root"):
            load_text_from_file("/etc/passwd", DELIMITER)

    def test_log_output_shows_relative_path(self, caplog):
        """Test that log output shows path relative to project root, not absolute"""
        file_content = f"{SAMPLE_STORY_1}{DELIMITER}\n"

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch.object(Path, "exists", return_value=True):
                load_text_from_file(TEST_FILE_PATH, DELIMITER)

        assert TEST_FILE_PATH in caplog.text
        assert "/Users" not in caplog.text
        assert f"Loading data from {TEST_FILE_PATH}" in caplog.text
