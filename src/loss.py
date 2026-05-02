"""
nanoLLM/src/loss.py

Cross-entropy loss function shared by training and future evaluation/perplexity scripts.
"""

import jax.numpy as jnp
import optax

from src.model.model import NanoLLM


def cross_entropy_loss(
    model: NanoLLM,
    batch: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Softmax cross-entropy with integer labels, mean-reduced.
    Returns (loss, logits)."""

    inputs, targets = batch

    # Feed the inputs to the model.
    # The results (logits) represent the denormalized likelihoods for each word in the dictionary that that word is next.
    logits = model(inputs)

    # Convert to probabilities (likely normalized)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    return loss, logits
