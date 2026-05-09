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


def compare_norms(state_before: nnx.State, state_after: nnx.State) -> NormsComparison:
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
    state_before: nnx.State,
    state_after: nnx.State,
    *,
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


def format_norms_comparison(report: NormsComparison) -> str:
    """Return a multi-line human-readable string for a NormsComparison result."""
    total_layers = report.compared_layers + report.skipped_layers

    percentage = abs(report.median_ratio - 1.0) * 100
    if percentage < 1.0:
        summary = f"The models are very similar. Median weight size differs by only {percentage:.2f}%."
    elif percentage < 5.0:
        summary = f"The models are moderately different. Median weight size differs by {percentage:.1f}%."
    else:
        direction = "grown" if report.median_ratio > 1.0 else "shrunk"
        summary = f"The models are noticeably different. Median weight size has {direction} by {percentage:.1f}%."

    lines = [
        "Weight magnitude (aka norm) comparison:",
        "",
        "  Remember this is a comparison of magnitude (not quality) and untrained models are designed to initialize with magnitudes similar to trained models.",
        "",
        "----------",
        "",
        f"  Median ratio:  {report.median_ratio:.4f}",
        "    This is the most representative ratio across all layers (robust to outliers).",
        "    Range: 0.0 to infinity",
        "      ~1.0   = weights are similar in scale",
        "      << 1.0 = weights shrank substantially overall",
        "      >> 1.0 = weights grew substantially overall",
        "",
        f"  Min ratio:     {report.min_ratio:.4f}",
        "    This is the ratio for the layer that shrank the most (or grew the least).",
        "    Range: 0.0 to infinity",
        "      ~0.0  = at least one layer nearly collapsed to zero (possible dead layer)",
        "      ~1.0  = even the most-shrunken layer barely changed in scale",
        "      > 1.0 = every layer grew; none shrank",
        "",
        f"  Max ratio:     {report.max_ratio:.4f}",
        "    This is the ratio for the layer that grew the most (or shrank the least).",
        "    Range: 0.0 to infinity",
        "      < 1.0  = every layer shrank; none grew",
        "      ~1.0   = even the most-grown layer barely changed in scale",
        "      >> 1.0 = at least one layer's weights grew dramatically",
        "",
        "----------",
        "",
        f"  {report.skipped_layers} layers (of {total_layers} total layers) were skipped due to presence of zero in the denominator.",
        f"  Total layers:   {total_layers}",
        f"  Compared:       {report.compared_layers}",
        f"  Skipped:        {report.skipped_layers}",
        "",
        "----------",
        "",
        f"SUMMARY: {summary}",
        "         NB: Models that have similar norms does not mean that they have identical weights. The states of the model before and after should be analyzed to examine parameters more closely.",
    ]
    return "\n".join(lines)


def format_state_comparison(report: StateComparison) -> str:
    """Return a multi-line human-readable string for a StateComparison result."""
    unchanged_params = report.total_params - report.changed_params

    if report.percent_changed < 1.0:
        summary = f"Almost no weights changed ({report.percent_changed:.2f}%). The two model states are nearly identical."
    elif report.percent_changed < 50.0:
        summary = f"A minority of weights changed ({report.percent_changed:.1f}%). Partial update or sparse fine-tuning likely."
    elif report.median_change < 1e-4:
        summary = f"{report.percent_changed:.1f}% of weights changed, but by very small amounts (median: {report.median_change:.2e}). The models are subtly different."
    else:
        summary = f"{report.percent_changed:.1f}% of weights changed, with a median shift of {report.median_change:.6f}. The models are substantially different."

    lines = [
        "Model parameter comparison:",
        "",
        "  Remember this compares actual weight values, not just magnitudes.",
        "  Values near zero indicate the model is unchanged in that respect.",
        "",
        "----------",
        "",
        f"  Percentage changed: {report.percent_changed:.2f}%",
        f"    Fraction of individual weight values that shifted by more than {report.change_threshold:g}.",
        "    Range: 0% to 100%",
        "      ~0%   = models are nearly identical (weights barely changed)",
        "      ~50%  = roughly half of all weights were updated",
        "      ~100% = almost every weight was updated",
        "",
        f"  Total parameters:   {report.total_params:,}",
        f"  Changed:            {report.changed_params:,}",
        f"  Unchanged:          {unchanged_params:,}",
        "",
        "----------",
        "",
        f"  Median absolute change: {report.median_change:.6f}",
        "    The most representative per-weight change (robust to outliers).",
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
        "      ~0.0  = at least one weight is essentially unchanged (expected if any params frozen)",
        "      > 0.0 = every single weight shifted by at least this amount",
        "",
        "----------",
        "",
        f"SUMMARY: {summary}",
        "         NB: A high percentage of changed weights is expected when loading a trained checkpoint into an untrained model.",
    ]
    return "\n".join(lines)


def state_snapshot(model: nnx.Module) -> nnx.State:
    """Return a deep copy of model's current state as a plain PyTree of arrays."""
    return jax.tree_util.tree_map(jnp.array, nnx.state(model))
