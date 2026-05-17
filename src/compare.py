"""
nanoLLM/src/compare.py

Checkpoint comparison utilities: compare weight magnitudes and parameter changes
between two model states.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

DEFAULT_CHANGE_THRESHOLD: float = 1e-8
MIN_REPORT_WIDTH: int = 40  # narrower than this breaks the closing-line dash prefix

NORMS_COMPARISON_INTRO: str = (
    " ANALYSIS 1 — NORMS (magnitude of weights)\n"
    " A shortcut comparison that measures the overall magnitude of each layer's weights, regardless of quality or direction.\n"
)

STATE_COMPARISON_INTRO: str = (
    " ANALYSIS 2 — STATE (actual value of all weights)\n"
    " Measures actual weight values and reports which fraction changed and by how much.\n"
)


@dataclass(frozen=True)
class NormsComparison:
    median_ratio: float
    min_ratio: float
    max_ratio: float
    compared_layers: int   # leaves with finite ratio
    skipped_layers: int    # leaves with zero/near-zero denominator (inf or NaN)


@dataclass(frozen=True)
class StateComparison:
    total_params: int
    changed_params: int
    percent_changed: float
    median_change: float
    mean_change: float
    max_change: float
    min_change: float
    change_threshold: float   # threshold used to determine if a parameter has "changed"


def compare_norms(*, state_before: nnx.State, state_after: nnx.State) -> NormsComparison:
    """Compare per-leaf L2-norm ratios between two model states.

    Arguments:
    If you are not loading directly from a checkpoint, ensure that the 'state' arguments
    are not references that could be mutated by subsequent training.
        Use state_snapshot to safely store deep copies of a state.


    Ratios = norm(after) / norm(before).
    Leaves with near-zero denominator (< 1e-12) are skipped because the ratio is infinitely positive or negative.

    Raises:
        ValueError: If state_before and state_after have different tree structures.
        This indicates that we are comparing two completely different models
        instead of one model in two different states of training.
    """
    struct_before = jax.tree_util.tree_structure(state_before)
    struct_after = jax.tree_util.tree_structure(state_after)
    if struct_before != struct_after:
        raise ValueError(
            f"state_before and state_after have different tree structures: "
            f"{struct_before} vs {struct_after}"
        )

    leaves_before = jax.tree_util.tree_leaves(state_before)
    leaves_after = jax.tree_util.tree_leaves(state_after)

    ratios: list[float] = []
    skipped = 0

    for leaf_before, leaf_after in zip(leaves_before, leaves_after):
        norm_before = float(jnp.linalg.norm(leaf_before.astype(float)))
        norm_after = float(jnp.linalg.norm(leaf_after.astype(float)))
        if norm_before < 1e-12:
            skipped += 1
        else:
            ratios.append(norm_after / norm_before)

    compared = len(ratios)
    if compared == 0:
        return NormsComparison(
            median_ratio=0.0,
            min_ratio=0.0,
            max_ratio=0.0,
            compared_layers=0,
            skipped_layers=skipped,
        )

    ratios_arr = jnp.array(ratios)
    return NormsComparison(
        median_ratio=float(jnp.median(ratios_arr)),
        min_ratio=float(jnp.min(ratios_arr)),
        max_ratio=float(jnp.max(ratios_arr)),
        compared_layers=compared,
        skipped_layers=skipped,
    )


def compare_states(
    *,
    state_before: nnx.State,
    state_after: nnx.State,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
) -> StateComparison:
    """Compare per-parameter absolute differences between two model states.

    Warning: The current state of this algorithm is not appropriate for
    very large models. It risks OOM when run on commercial models.

    Arguments:
    If you are not loading directly from a checkpoint, ensure that the 'state' arguments
    are not references that could be mutated by subsequent training.
    Use state_snapshot to safely store deep copies of a state.

    Raises:
        ValueError: If state_before and state_after have different tree structures.
    """
    struct_before = jax.tree_util.tree_structure(state_before)
    struct_after = jax.tree_util.tree_structure(state_after)
    if struct_before != struct_after:
        raise ValueError(
            f"state_before and state_after have different tree structures: "
            f"{struct_before} vs {struct_after}"
        )

    leaves_before = jax.tree_util.tree_leaves(state_before)
    leaves_after = jax.tree_util.tree_leaves(state_after)

    # TODO: Refactor to derisk OOM on very large models
    # This materializes the full diff array (potentially hundreds of millions of floats for larger models).
    # e.g., A model with 100M parameters would consume roughly 400MB for this comparison.
    # Potential improvement: Compute statistics per-leaf and aggregate (e.g., mean-of-means) instead of concatenating all figures.
    diffs = jnp.concatenate(
        [
            jnp.abs(a.astype(float) - b.astype(float)).ravel()
            for a, b in zip(leaves_before, leaves_after)
        ]
    )

    total_params = int(diffs.size)
    changed_params = int(jnp.sum(diffs > threshold))
    percent_changed = (changed_params / total_params) * 100.0
    median_change = float(jnp.median(diffs))
    mean_change = float(jnp.mean(diffs))
    max_change = float(jnp.max(diffs))
    min_change = float(jnp.min(diffs))

    return StateComparison(
        total_params=total_params,
        changed_params=changed_params,
        percent_changed=percent_changed,
        median_change=median_change,
        mean_change=mean_change,
        max_change=max_change,
        min_change=min_change,
        change_threshold=threshold,
    )


def format_norms_comparison(report: NormsComparison, width: int) -> str:
    """Return a multi-line human-readable string for a NormsComparison result."""
    if width < MIN_REPORT_WIDTH:
        raise ValueError(f"width must be at least {MIN_REPORT_WIDTH}, got {width}")

    div: str = "-" * width
    heavy_div: str = "=" * width
    closing: str = "end of norm analysis"
    closing_size: int = len(closing)
    total_layers = report.compared_layers + report.skipped_layers

    # construct summary
    percentage = abs(report.median_ratio - 1.0) * 100
    if percentage < 1.0:
        summary = f"The models are very similar. Median weight size differs by only {percentage:.2f}%."
    elif percentage < 5.0:
        summary = f"The models are moderately different. Median weight size differs by {percentage:.1f}%."
    else:
        direction = "grown" if report.median_ratio > 1.0 else "shrunk"
        summary = f"The models are noticeably different. Median weight size has {direction} by {percentage:.1f}%."

    lines = [
        heavy_div,
        " ANALYSIS 1: WEIGHT MAGNITUDE (NORMS)",
        heavy_div,
        "",
        f" RESULT SUMMARY: {summary}",
        "",
        div,
        " CONTEXT",
        "",
        "  This analysis is a shortcut comparison, measuring the overall magnitude of each layer's weights.",
        "  It does not indicate difference in quality or direction (positive/negative) of the weights.",
        "",
        "  A ratio near 1.0 means similar scale, not similar weights.",
        "         1.0 = same scale  |  2.0 = after is twice as large  |  0.5 = half the size",
        "",
        "  Each layer contains hundreds or thousands of individual weights.",
        "  The norm of a layer collapses all of those weights into a single number that represents the overall size (magnitude) of the weights.",
        "  So comparing the norms of the layers of two checkpoints necessarily ignores a level of detail.",
        "  See Analysis of State for a more definitive measure of difference between checkpoints.",
        "",
        "   NB: Well-constructed models are designed to initialize weights at the same scale as a trained model.",
        "       So even an untrained model might exhibit a very small norm diff compared to a trained model.",
        "",
        div,
        " RATIOS",
        "",
        f"  Median ratio:  {report.median_ratio:.4f}",
        "    Most representative ratio across all layers (robust to outliers).",
        "    Range: 0.0 to infinity",
        "      ~1.0   = weights are similar in scale",
        "      << 1.0 = weights shrank substantially overall",
        "      >> 1.0 = weights grew substantially overall",
        "",
        f"  Min ratio:     {report.min_ratio:.4f}",
        "    Ratio for the layer that shrank the most (or grew the least).",
        "    Range: 0.0 to infinity",
        "      ~0.0  = at least one layer nearly collapsed to zero (possible dead layer)",
        "      ~1.0  = even the most-shrunken layer barely changed in scale",
        "      > 1.0 = every layer grew; none shrank",
        "",
        f"  Max ratio:     {report.max_ratio:.4f}",
        "    Ratio for the layer that grew the most (or shrank the least).",
        "    Range: 0.0 to infinity",
        "      < 1.0  = every layer shrank; none grew",
        "      ~1.0   = even the most-grown layer barely changed in scale",
        "      >> 1.0 = at least one layer's weights grew dramatically",
        "",
        div,
        " LAYER COUNTS",
        "",
        f"  {report.skipped_layers} layers (of {total_layers} total layers) were skipped due to presence of zero in the denominator.",
        f"  Total layers:   {total_layers}",
        f"  Compared:       {report.compared_layers}",
        f"  Skipped:        {report.skipped_layers}",
        "",
        "",
        f"{'-' * (width - closing_size)}{closing}",
        "",
    ]
    return "\n".join(lines)


def format_state_comparison(report: StateComparison, width: int) -> str:
    """Return a multi-line human-readable string for a StateComparison result."""
    if width < MIN_REPORT_WIDTH:
        raise ValueError(f"width must be at least {MIN_REPORT_WIDTH}, got {width}")

    div: str = "-" * width
    heavy_div: str = "=" * width
    closing: str = "end of state analysis"
    closing_size: int = len(closing)
    unchanged_params = report.total_params - report.changed_params

    # construct summary
    if report.percent_changed < 1.0:
        summary = f"Almost no weights changed ({report.percent_changed:.2f}%). The two model states are nearly identical."
    elif report.percent_changed < 50.0:
        summary = f"A minority of weights changed ({report.percent_changed:.1f}%). Partial update or sparse fine-tuning likely."
    elif report.median_change < 1e-4:
        summary = f"{report.percent_changed:.1f}% of weights changed, but by very small amounts (median: {report.median_change:.2e}). The models are subtly different."
    else:
        summary = f"{report.percent_changed:.1f}% of weights changed, with a median shift of {report.median_change:.6f}. The models are substantially different."

    lines = [
        heavy_div,
        " ANALYSIS 2: WEIGHT VALUES (STATE COMPARISON)",
        heavy_div,
        "",
        f" RESULT SUMMARY: {summary}",
        "",
        div,
        " CONTEXT",
        "",
        "  This analysis compares the actual weight values (potentially millions of values).",
        "  It reports the fraction of weights that changed and by how much.",
        "",
        "  Values near zero in this analysis indicate the model is unchanged.",
        "  Unlike norms analysis, a high percentage of change in state analysis is expected when comparing a trained checkpoint against an untrained model.",
        "",
        div,
        " CHANGE RATE",
        "",
        f"  Percentage changed: {report.percent_changed:.2f}%",
        f"    Fraction of individual weights that shifted by more than {report.change_threshold:g}.",
        "    Range: 0% to 100%",
        "      ~0%   = models are nearly identical (weights barely changed)",
        "      ~50%  = roughly half of all weights were updated",
        "      ~100% = almost every weight was updated",
        "",
        f"  Total parameters:   {report.total_params:,}",
        f"  Changed:            {report.changed_params:,}",
        f"  Unchanged:          {unchanged_params:,}",
        "",
        div,
        " CHANGE MAGNITUDES",
        "",
        f"  Median absolute change: {report.median_change:.6f}",
        "    Most representative per-weight change (robust to outliers).",
        "    Range: 0.0 to infinity",
        "      ~0.0  = the typical weight barely moved",
        "      small = fine-grained updates; weights shifted only slightly on average",
        "      large = weights shifted substantially on average",
        "",
        f"  Mean absolute change:   {report.mean_change:.6f}",
        "    Average per-weight change across all parameters.",
        "    Range: 0.0 to infinity",
        "      ~0.0       = weights are nearly identical overall",
        "      >> median  = a few large outlier updates are skewing the average",
        "      ~ median   = changes are uniformly distributed across weights",
        "",
        f"  Max absolute change:    {report.max_change:.6f}",
        "    The single largest weight shift anywhere in the model.",
        "    Range: 0.0 to infinity",
        "      ~0.0  = no weight changed significantly",
        "      large = at least one weight shifted dramatically (may indicate an outlier layer)",
        "",
        f"  Min absolute change:    {report.min_change:.6f}",
        "    The smallest weight shift (includes unchanged weights at ~0.0).",
        "    Range: 0.0 to infinity",
        "      ~0.0  = at least one weight is essentially unchanged",
        "      > 0.0 = every single weight shifted by at least this amount",
        "",
        "",
        f"{'-' * (width - closing_size)}{closing}",
        "",
    ]
    return "\n".join(lines)


def state_snapshot(model: nnx.Module) -> nnx.State:
    """Return a deep copy of model's current state as a plain PyTree of arrays."""
    return jax.tree_util.tree_map(jnp.array, nnx.state(model))
