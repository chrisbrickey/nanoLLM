"""Unit tests for src/logging_setup.py"""

import logging

import pytest

from src.logging_setup import setup_logging


class TestSetupLogging:
    def test_accepts_uppercase_level(self) -> None:
        setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_accepts_lowercase_level(self) -> None:
        setup_logging("debug")
        assert logging.getLogger().level == logging.DEBUG

    @pytest.mark.parametrize("level", ["INVALID", "verbose", "trace", ""])
    def test_rejects_invalid_level(self, level: str) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level)
