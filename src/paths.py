from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DEFAULT_DATA_FILE = DATA_RAW_DIR / "TinyStories-1000.txt"
# DATA_PROCESSED_DIR = DATA_DIR / "processed" # future feature for processing larger datasets


def validate_project_path(file_path: str | Path) -> Path:
    """
    Validate that a file path is within the project root.

    Converts relative paths to absolute paths relative to PROJECT_ROOT
    and ensures the resolved path does not escape the project directory.

    Args:
        file_path: Path to validate (can be str or Path, relative or absolute)

    Returns:
        Resolved absolute Path object within the project root

    Raises:
        ValueError: If the path attempts to escape the project root

    Examples:
        >>> validate_project_path("data/file.txt")  # OK
        >>> validate_project_path("../../../etc/passwd")  # Raises ValueError
    """
    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    resolved_path = path.resolve()

    try:
        resolved_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"Path '{file_path}' resolves to '{resolved_path}' which is outside "
            f"the project root '{PROJECT_ROOT.resolve()}'. All file paths must "
            f"stay within the project directory."
        )

    return resolved_path


def format_path_for_display(file_path: str | Path) -> Path:
    """
    Format a file path for display by showing it relative to PROJECT_ROOT.

    This ensures paths are displayed without exposing the full system path,
    improving privacy and readability.

    Args:
        file_path: Path to format (can be str or Path, relative or absolute)

    Returns:
        Path object relative to PROJECT_ROOT for display

    Examples:
        >>> format_path_for_display("/Users/name/project/data/file.txt")
        PosixPath('data/file.txt')
        >>> format_path_for_display("data/file.txt")
        PosixPath('data/file.txt')
    """
    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return Path(path.name)
