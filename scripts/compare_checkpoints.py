"""
nanoLLM/scripts/compare_checkpoints.py

CLI entry point for comparing two nanoLLM checkpoints.

Usage:
    uv run nanollm-compare                                      # defaults to two most recent
    uv run nanollm-compare --before path/A --after path/B       # explicit
    uv run nanollm-compare --threshold 1e-6                     # custom "changed" cutoff
"""

import argparse
import logging
import sys
from pathlib import Path

import flax.nnx as nnx

from src.checkpoint import build_model_from_checkpoint, get_latest_checkpoints
from src.compare import (
    DEFAULT_CHANGE_THRESHOLD,
    NORMS_COMPARISON_INTRO,
    STATE_COMPARISON_INTRO,
    compare_norms,
    compare_states,
    format_norms_comparison,
    format_state_comparison,
    NormsComparison,
    StateComparison,
)
from src.logging_setup import setup_logging
from src.paths import CHECKPOINTS_DIR, validate_project_path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two nanoLLM checkpoints.")
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help="Path to the 'before' checkpoint bundle",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Path to the 'after' checkpoint bundle",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CHANGE_THRESHOLD,
        help=f"Threshold for counting a parameter as 'changed' (default: {DEFAULT_CHANGE_THRESHOLD})",
    )
    args = parser.parse_args()
    if args.threshold <= 0:
        parser.error(f"--threshold must be positive; got {args.threshold}")
    return args


def _terminal_printout(norms_report: NormsComparison, states_report: StateComparison) -> None:

    #--- introductory section ---

    separator_count: int = 90
    separator: str = "#" * separator_count
    title: str = "CHECKPOINT DIFF ANALYSIS RESULTS"

    banner = (
        "\n\n"
        f"{separator}\n\n"
        f" {title}\n\n"
        f"{separator}\n\n"
    )

    print(
        f"{banner}"
        f"\n Multiple analyses follow:\n\n"
        f"{NORMS_COMPARISON_INTRO}\n\n"
        f"{STATE_COMPARISON_INTRO}\n\n"
    )

    #-----------------------------------

    print(f"\n{format_norms_comparison(norms_report, separator_count)}")
    print(f"\n{format_state_comparison(states_report, separator_count)}")
    print(f"\n{separator}\n\n")


def main() -> None:
    setup_logging()
    args = _parse_args()

    try:
        # Resolve paths
        if (args.before is None) != (args.after is None):
            raise ValueError("Provide both --before and --after, or neither.")

        if args.before is None and args.after is None:
            checkpoints = get_latest_checkpoints(CHECKPOINTS_DIR, n=2)
            if len(checkpoints) < 2:
                raise ValueError(
                    f"Need exactly 2 checkpoints in {CHECKPOINTS_DIR}; found {len(checkpoints)}."
                )
            # get_latest_checkpoints returns newest-first; before=older, after=newer
            path_after, path_before = checkpoints[0], checkpoints[1]
        else:
            path_before = validate_project_path(Path(args.before))
            path_after = validate_project_path(Path(args.after))

        # Warn if same path
        if path_before == path_after:
            logger.warning(
                "--before and --after resolve to the same path; report will show 0%% changed."
            )

        logger.info(
            "\n\n%s\nComparing checkpoints:\n\tbefore: %s\n\tafter:  %s\n%s\n\n",
            "-" * 30,
            path_before,
            path_after,
            "-" * 30,
        )

        # Load both checkpoints
        try:
            model_before, _, _ = build_model_from_checkpoint(path_before)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(
                f"Failed to load 'before' checkpoint at '{path_before}': {e}"
            ) from e

        try:
            model_after, _, _ = build_model_from_checkpoint(path_after)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(
                f"Failed to load 'after' checkpoint at '{path_after}': {e}"
            ) from e

        state_before = nnx.state(model_before)
        state_after = nnx.state(model_after)

        logger.info("Running checkpoint diff analysis. This may take a few minutes depending on the size of the model.")

        norms_report = compare_norms(state_before, state_after)
        states_report = compare_states(state_before, state_after, threshold=args.threshold)

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error("%s", e)
        sys.exit(1)

    # Print out report
    _terminal_printout(norms_report, states_report)


if __name__ == "__main__":
    main()
