"""
Central configuration for nanoLLM project.

This module defines all paths, model hyperparameters, and training configurations
used throughout the project. Import from this module to ensure consistency across
notebooks and source code.
"""

from pathlib import Path
from dataclasses import dataclass
import tiktoken

# ============================================================================
# Project Paths
# ============================================================================

# Project root (src/../ = project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure necessary directories exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Data file paths
TINYSTORIES_FILE = DATA_DIR / "TinyStories-1000.txt"


# ============================================================================
# Path Validation Utilities
# ============================================================================

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

    # If path is relative, make it relative to PROJECT_ROOT
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Resolve to absolute path (eliminates .., ., symlinks)
    resolved_path = path.resolve()

    # Check if resolved path is within PROJECT_ROOT
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

    # Resolve to absolute path if needed
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Try to make it relative to PROJECT_ROOT
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        # If not within project root, just return the filename
        return Path(path.name)


# ============================================================================
# Tokenizer Configuration
# ============================================================================

@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for tokenizer."""
    delimiter: str  # Required: delimiter string for the data source
    name: str  # Required: tokenizer name (e.g., "gpt2", "gpt4", etc.)

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        """Get the tokenizer instance."""
        return tiktoken.get_encoding(self.name)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.n_vocab

    @property
    def end_token(self) -> str:
        """Get the end-of-text token (alias for delimiter)."""
        return self.delimiter


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for NanoLLM model architecture."""

    # Sequence and vocabulary
    maxlen: int = 128
    vocab_size: int = 50257  # GPT-2 tokenizer default

    # Embedding dimensions
    embed_dim: int = 192

    # Attention configuration
    num_heads: int = 6

    # Feed-forward dimension (default: 2/3 * 4 * embed_dim)
    feed_forward_dim: int = 512

    # Number of transformer blocks
    num_transformer_blocks: int = 6

    # Random seed for model initialization
    model_seed: int = 0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Data loading
    batch_size: int = 32
    max_stories: int = 100
    shuffle: bool = False
    seed: int = 42
    num_workers: int = 0  # Single process for experimentation

    # Training loop
    num_epochs: int = 3

    # Learning rate schedule
    lr_init_value: float = 0.0
    lr_peak_value: float = 3e-4
    lr_end_value: float = 1e-5
    warmup_rate: float = 0.1  # 10% of total steps

    # Optimizer
    weight_decay: float = 0.01

    # Logging
    log_every_n_steps: int = 2

    def calculate_training_steps(self, batches_per_epoch: int) -> tuple[int, int]:
        """
        Calculate total training steps and warmup steps.

        Args:
            batches_per_epoch: Number of batches per epoch

        Returns:
            Tuple of (total_steps, warmup_steps)
        """
        total_steps = batches_per_epoch * self.num_epochs
        warmup_steps = max(1, int(total_steps * self.warmup_rate))
        return total_steps, warmup_steps


# ============================================================================
# Default Configuration Instances
# ============================================================================

# Create default instances for easy import
model_config = ModelConfig()
training_config = TrainingConfig()