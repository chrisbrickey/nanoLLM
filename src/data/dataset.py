"""Dataset class for story data preprocessing."""

from src.config import TokenizerConfig


class StoryDataset:
    """
    Dataset for tokenizing and padding story text.

    This class handles tokenization, truncation, and padding to prepare
    stories for training.
    """

    def __init__(
        self,
        stories: list[str],
        maxlen: int,
        tokenizer_config: TokenizerConfig,
    ) -> None:
        """
        Initialize dataset.

        Args:
            stories: List of story strings
            maxlen: Maximum sequence length (for truncation and padding)
            tokenizer_config: Source of tokenizer, delimiter, and pad_token_id
        """
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer_config.tokenizer
        self.delimiter = tokenizer_config.delimiter
        self.pad_token_id = tokenizer_config.pad_token_id

    def __len__(self) -> int:
        """Return number of stories in dataset."""
        return len(self.stories)

    def __getitem__(self, idx: int) -> list[int]:
        """
        Get tokenized and padded story at index.

        Args:
            idx: Index of story to retrieve

        Returns:
            List of token IDs, truncated and padded to maxlen
        """
        story = self.stories[idx]
        tokens = self.tokenizer.encode(
            story, allowed_special={self.delimiter}
        )

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend(
            [self.pad_token_id] * (self.maxlen - len(tokens))
        )
        return tokens
