from dataclasses import dataclass
import tiktoken


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
