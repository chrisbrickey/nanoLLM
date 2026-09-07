from src.config import InferenceConfig, TokenizerConfig
from src.inference.generate import generate_text
from src.model.model import NanoLLM


def complete_prompt(
    *,
    model: NanoLLM,
    tokenizer_config: TokenizerConfig,
    inference_config: InferenceConfig,
    prompt: str,
) -> str:
    """Encode a text prompt and generate a completion for it.

    Raises:
        ValueError: If prompt is empty or whitespace-only.
    """
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    start_tokens = tokenizer_config.tokenizer.encode(prompt)
    return generate_text(
        model=model,
        tokenizer_config=tokenizer_config,
        inference_config=inference_config,
        start_tokens=start_tokens,
    )
