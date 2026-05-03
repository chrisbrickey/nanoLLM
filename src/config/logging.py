"""Logging configuration for nanoLLM."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Call once at the application entry point."""
    numeric_level = getattr(logging, level.upper())
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove absl handlers so basicConfig can install nanoLLM logging without
    # evicting other handlers such as pytest's caplog handler.
    # Why? absl-py (imported by JAX at load time) installs its own handler on the
    # root logger before setup_logging() runs, making basicConfig a no-op.
    # I want to avoid coupling nanoLLM logging to a JAX implementation dependency.
    # If JAX ever drops or changes absl, I do not want nanoLLM logging to break.
    for handler in list(root.handlers):
        if type(handler).__module__.startswith("absl"):
            root.removeHandler(handler)

    # Supress absl logging to reduce noisiness in logs, especially during training
    logging.getLogger("absl").setLevel(logging.WARNING)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
