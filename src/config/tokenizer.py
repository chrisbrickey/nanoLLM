from dataclasses import dataclass
import tiktoken


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for tokenizer. Defaults target gpt2 with the standard end-of-text marker.

    The tokenizer name and delimiter are stored together in TokenizerConfig because they are tightly coupled
    during training. Together, they describe how a model was trained, which is why this metadata must
    be stored and loaded rom checkpoints. Neither tokenizer nor delimeter should be coupled directly to the
    model because they might change if the model is retrained.

    The delimiter serves different purposes during training vs inference
        During training: Marks boundaries between stories in your dataset
        During inference: Acts as a stop condition. If the model generates the end token, it means the paragraph is likely complete.
    """
    delimiter: str = "<|endoftext|>"
    name: str = "gpt2"
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if not self.delimiter:
            raise ValueError("delimiter must be a non-empty string")
        if self.pad_token_id < 0:
            raise ValueError(f"pad_token_id must be >= 0, got {self.pad_token_id}")
        try:
            tiktoken.get_encoding(self.name)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Unknown tiktoken encoding name: {self.name!r} ({e})") from e

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        """Get the tokenizer instance."""
        return tiktoken.get_encoding(self.name)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (aka the size of the encoding scheme).
        This is a property of the algorithm/model - the total number of tokens that the encoding can represent. e.g., GPT-2 has 50,257.
        vocab_size here is NOT the tokens that are actually observed in a corpus."""
        return self.tokenizer.n_vocab

    @property
    def end_token(self) -> str:
        """Get the end-of-text token (alias for delimiter)."""
        return self.delimiter
