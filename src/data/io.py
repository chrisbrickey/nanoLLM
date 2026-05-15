"""File I/O for loading delimited text from disk.

Pure I/O helpers — no tokenization, no batching. Consumers convert the
returned raw text blocks into model-ready batches via src.data.processor.
"""

import logging
from pathlib import Path

from src.paths import validate_project_path, format_path_for_display

logger = logging.getLogger(__name__)


def load_text_from_file(
    *,
    file_path: str | Path,
    delimiter: str,
    max_paragraphs: int | None = None,
) -> list[str]:
    """Efficiently loads paragraphs from a text file.

    Paragraphs are loaded line by line to avoid loading the entire file into memory.

    Args:
        file_path: Path to the text file
        delimiter: String that marks the end of a paragraph
        max_paragraphs: Maximum number of paragraphs to load (None for all)

    Returns:
        List of stripped text strings, one per delimited paragraph

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the path attempts to escape the project root
    """

    file_path = validate_project_path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    display_path = format_path_for_display(file_path)
    max_str = f"{max_paragraphs:,}" if max_paragraphs is not None else "all"
    logger.info(f"Loading data from {display_path} (max {max_str} paragraphs)")

    paragraphs, current_paragraph = [], []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if delimiter in line:
                parts = line.split(delimiter)
                for part in parts[:-1]:
                    current_paragraph.append(part)
                    story_text = ''.join(current_paragraph).strip()
                    if story_text:
                        paragraphs.append(story_text + delimiter)
                        if max_paragraphs and len(paragraphs) >= max_paragraphs:
                            break
                    current_paragraph = []

                if parts[-1].strip():
                    current_paragraph = [parts[-1]]
                else:
                    current_paragraph = []

                if max_paragraphs and len(paragraphs) >= max_paragraphs:
                    break
            else:
                current_paragraph.append(line)

        if current_paragraph and (not max_paragraphs or len(paragraphs) < max_paragraphs):
            story_text = ''.join(current_paragraph).strip()
            if story_text:
                paragraphs.append(story_text + delimiter)

    logger.info(f"Loaded {len(paragraphs)} paragraphs")
    return paragraphs
