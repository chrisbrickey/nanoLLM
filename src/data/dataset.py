"""Dataset class for story data preprocessing."""

import tiktoken


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
        delimiter: str,
        tokenizer_name: str = "gpt2"
    ) -> None:
        """
        Initialize dataset.

        Args:
            stories: List of story strings
            maxlen: Maximum sequence length (for truncation and padding)
            delimiter: Delimiter string to mark end of text
            tokenizer_name: Name of tokenizer to use (default: "gpt2")
        """
        self.stories = stories
        self.maxlen = maxlen
        self.delimiter = delimiter
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.delimiter_token = self.tokenizer.encode(self.delimiter, allowed_special={self.delimiter})[0]

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
        tokens = self.tokenizer.encode(story, allowed_special={self.delimiter})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens
