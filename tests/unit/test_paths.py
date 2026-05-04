"""Unit tests for src/paths.py path utilities."""

from pathlib import Path

import pytest

from src.paths import PROJECT_ROOT, format_path_for_display, validate_project_path


class TestValidateProjectPath:
    """Test suite for validate_project_path utility function"""

    def test_relative_path_within_project(self) -> None:
        result = validate_project_path("data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_absolute_path_within_project(self) -> None:
        test_path = PROJECT_ROOT / "data" / "test.txt"
        result = validate_project_path(str(test_path))
        assert result == test_path.resolve()

    def test_path_object_input(self) -> None:
        test_path = Path("data/test.txt")
        result = validate_project_path(test_path)
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_nested_relative_path(self) -> None:
        result = validate_project_path("data/raw/stories.txt")
        expected = (PROJECT_ROOT / "data/raw/stories.txt").resolve()
        assert result == expected

    def test_parent_directory_escape_raises_error(self) -> None:
        with pytest.raises(ValueError, match="outside the project root"):
            validate_project_path("../../etc/passwd")

    def test_absolute_path_outside_project_raises_error(self) -> None:
        with pytest.raises(ValueError, match="outside the project root"):
            validate_project_path("/etc/passwd")

    def test_current_directory_notation(self) -> None:
        result = validate_project_path("./data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected

    def test_project_root_itself(self) -> None:
        result = validate_project_path(".")
        assert result == PROJECT_ROOT.resolve()

    def test_path_with_multiple_parent_refs(self) -> None:
        result = validate_project_path("data/../data/test.txt")
        expected = (PROJECT_ROOT / "data/test.txt").resolve()
        assert result == expected


class TestFormatPathForDisplay:
    """Test suite for format_path_for_display utility function"""

    def test_absolute_path_within_project(self) -> None:
        test_path = PROJECT_ROOT / "data" / "sample.txt"
        result = format_path_for_display(test_path)
        assert result == Path("data/sample.txt")

    def test_relative_path_stays_relative(self) -> None:
        result = format_path_for_display("data/sample.txt")
        assert result == Path("data/sample.txt")

    def test_path_outside_project_shows_filename_only(self) -> None:
        result = format_path_for_display("/tmp/external.txt")
        assert result == Path("external.txt")
