"""Unit tests for src/compare.py"""

import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from src.compare import (
    DEFAULT_CHANGE_THRESHOLD,
    MIN_REPORT_WIDTH,
    NormsComparison,
    StateComparison,
    compare_norms,
    compare_states,
    format_norms_comparison,
    format_state_comparison,
    state_snapshot,
)
from src.config import ModelConfig
from src.model.model import NanoLLM

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1


def _make_model(seed: int = 0) -> NanoLLM:
    config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
        model_seed=seed,
    )
    return NanoLLM(config)


@pytest.fixture
def identical_states() -> tuple[dict, dict]:
    state = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0, 4.0])}
    return state, state


@pytest.fixture
def doubled_state() -> tuple[dict, dict]:
    """State B has every value exactly 2× state A."""
    state_a = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0, 4.0])}
    state_b = {"layer1": jnp.array([2.0, 4.0]), "layer2": jnp.array([6.0, 8.0])}
    return state_a, state_b


# ---------------------------------------------------------------------------
# TestCompareNorms
# ---------------------------------------------------------------------------


class TestCompareNorms:
    def test_identical_states_gives_ratios_of_one(self, identical_states) -> None:
        state_a, state_b = identical_states
        result = compare_norms(state_a, state_b)

        assert result.median_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.min_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.max_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.skipped_layers == 0

    def test_doubled_state_gives_ratios_of_two(self, doubled_state) -> None:
        state_a, state_b = doubled_state
        result = compare_norms(state_a, state_b)

        assert result.median_ratio == pytest.approx(2.0, abs=1e-6)
        assert result.min_ratio == pytest.approx(2.0, abs=1e-6)
        assert result.max_ratio == pytest.approx(2.0, abs=1e-6)
        assert result.skipped_layers == 0

    def test_zero_denominator_leaf_counted_in_skipped(self) -> None:
        state_a = {"layer1": jnp.array([0.0, 0.0]), "layer2": jnp.array([1.0, 2.0])}
        state_b = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([2.0, 4.0])}

        result = compare_norms(state_a, state_b)

        assert result.skipped_layers == 1
        assert result.compared_layers == 1

    def test_all_zero_after_gives_ratios_of_zero(self) -> None:
        state_a = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0, 4.0])}
        state_b = {"layer1": jnp.array([0.0, 0.0]), "layer2": jnp.array([0.0, 0.0])}

        result = compare_norms(state_a, state_b)

        assert result.median_ratio == pytest.approx(0.0, abs=1e-6)
        assert result.min_ratio == pytest.approx(0.0, abs=1e-6)
        assert result.max_ratio == pytest.approx(0.0, abs=1e-6)
        assert result.skipped_layers == 0

    def test_very_small_denominator_triggers_epsilon_floor_and_is_skipped(self) -> None:
        tiny = 1e-15
        state_a = {"layer1": jnp.array([tiny, tiny]), "layer2": jnp.array([1.0, 2.0])}
        state_b = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([2.0, 4.0])}

        result = compare_norms(state_a, state_b)

        # The tiny-denominator leaf must be skipped
        assert result.skipped_layers >= 1

    def test_mismatched_pytree_keys_raises_value_error(self) -> None:
        state_a = {"layer1": jnp.array([1.0, 2.0])}
        state_b = {"layer_different": jnp.array([1.0, 2.0])}

        with pytest.raises(ValueError):
            compare_norms(state_a, state_b)


# ---------------------------------------------------------------------------
# TestCompareStates
# ---------------------------------------------------------------------------


class TestCompareStates:
    def test_identical_states_gives_zero_change(self, identical_states) -> None:
        state_a, state_b = identical_states
        result = compare_states(state_a, state_b)

        assert result.percent_changed == pytest.approx(0.0, abs=1e-6)
        assert result.median_change == pytest.approx(0.0, abs=1e-6)
        assert result.mean_change == pytest.approx(0.0, abs=1e-6)
        assert result.max_change == pytest.approx(0.0, abs=1e-6)
        assert result.min_change == pytest.approx(0.0, abs=1e-6)

    def test_completely_different_states_gives_nearly_full_change(self) -> None:
        state_a = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0, 4.0])}
        state_b = {"layer1": jnp.array([100.0, 200.0]), "layer2": jnp.array([300.0, 400.0])}

        result = compare_states(state_a, state_b)

        assert result.percent_changed == pytest.approx(100.0, abs=1e-3)

    def test_mixed_change_gives_intermediate_percent(self) -> None:
        # 2 out of 4 params change
        state_a = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0, 4.0])}
        state_b = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([300.0, 400.0])}

        result = compare_states(state_a, state_b)

        assert 0.0 < result.percent_changed < 100.0

    def test_threshold_respected_small_diff_not_counted(self) -> None:
        small_delta = 1e-10  # well under the default 1e-8
        state_a = {"layer1": jnp.array([1.0, 2.0])}
        state_b = {"layer1": jnp.array([1.0 + small_delta, 2.0 + small_delta])}

        result = compare_states(state_a, state_b, threshold=DEFAULT_CHANGE_THRESHOLD)

        assert result.changed_params == 0
        assert result.percent_changed == pytest.approx(0.0, abs=1e-6)

    def test_threshold_respected_large_diff_is_counted(self) -> None:
        large_delta = 1.0
        state_a = {"layer1": jnp.array([1.0, 2.0])}
        state_b = {"layer1": jnp.array([1.0 + large_delta, 2.0 + large_delta])}

        result = compare_states(state_a, state_b, threshold=DEFAULT_CHANGE_THRESHOLD)

        assert result.changed_params == 2

    def test_threshold_parameter_stored_in_result(self) -> None:
        custom_threshold = 1e-4
        state_a = {"layer1": jnp.array([1.0])}
        state_b = {"layer1": jnp.array([2.0])}

        result = compare_states(state_a, state_b, threshold=custom_threshold)

        assert result.change_threshold == pytest.approx(custom_threshold)

    def test_mismatched_structure_raises_value_error(self) -> None:
        state_a = {"layer1": jnp.array([1.0, 2.0])}
        state_b = {"layer_different": jnp.array([1.0, 2.0])}

        with pytest.raises(ValueError):
            compare_states(state_a, state_b)


# ---------------------------------------------------------------------------
# TestFormatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_norms_comparison_contains_key_values(self) -> None:
        report = NormsComparison(
            median_ratio=1.5,
            min_ratio=0.8,
            max_ratio=2.2,
            compared_layers=4,
            skipped_layers=1,
        )
        output = format_norms_comparison(report, width=80)

        assert "1.5" in output or "1.50" in output
        assert "0.8" in output or "0.80" in output
        assert "2.2" in output or "2.20" in output

    def test_format_state_comparison_contains_key_values(self) -> None:
        report = StateComparison(
            total_params=100,
            changed_params=42,
            percent_changed=42.0,
            median_change=0.01,
            mean_change=0.02,
            max_change=0.5,
            min_change=0.0,
            change_threshold=1e-8,
        )
        output = format_state_comparison(report, width=80)

        assert "100" in output
        assert "42" in output

    def test_format_norms_comparison_with_zero_compared_layers_does_not_crash(self) -> None:
        report = NormsComparison(
            median_ratio=0.0,
            min_ratio=0.0,
            max_ratio=0.0,
            compared_layers=0,
            skipped_layers=3,
        )
        # Must not raise ZeroDivisionError and must not produce "nan" in output
        output = format_norms_comparison(report, width=80)

        assert "nan" not in output.lower()

    def test_format_norms_comparison_raises_on_width_below_minimum(self) -> None:
        report = NormsComparison(
            median_ratio=1.0, min_ratio=1.0, max_ratio=1.0,
            compared_layers=1, skipped_layers=0,
        )
        with pytest.raises(ValueError):
            format_norms_comparison(report, width=MIN_REPORT_WIDTH - 1)

    def test_format_norms_comparison_raises_on_negative_width(self) -> None:
        report = NormsComparison(
            median_ratio=1.0, min_ratio=1.0, max_ratio=1.0,
            compared_layers=1, skipped_layers=0,
        )
        with pytest.raises(ValueError):
            format_norms_comparison(report, width=-1)

    def test_format_norms_comparison_accepts_minimum_width(self) -> None:
        report = NormsComparison(
            median_ratio=1.0, min_ratio=1.0, max_ratio=1.0,
            compared_layers=1, skipped_layers=0,
        )
        output = format_norms_comparison(report, width=MIN_REPORT_WIDTH)
        assert len(output) > 0

    def test_format_state_comparison_raises_on_width_below_minimum(self) -> None:
        report = StateComparison(
            total_params=10, changed_params=0, percent_changed=0.0,
            median_change=0.0, mean_change=0.0, max_change=0.0,
            min_change=0.0, change_threshold=1e-8,
        )
        with pytest.raises(ValueError):
            format_state_comparison(report, width=MIN_REPORT_WIDTH - 1)

    def test_format_state_comparison_raises_on_negative_width(self) -> None:
        report = StateComparison(
            total_params=10, changed_params=0, percent_changed=0.0,
            median_change=0.0, mean_change=0.0, max_change=0.0,
            min_change=0.0, change_threshold=1e-8,
        )
        with pytest.raises(ValueError):
            format_state_comparison(report, width=-1)

    def test_format_state_comparison_accepts_minimum_width(self) -> None:
        report = StateComparison(
            total_params=10, changed_params=0, percent_changed=0.0,
            median_change=0.0, mean_change=0.0, max_change=0.0,
            min_change=0.0, change_threshold=1e-8,
        )
        output = format_state_comparison(report, width=MIN_REPORT_WIDTH)
        assert len(output) > 0

    def test_format_state_comparison_mentions_threshold(self) -> None:
        threshold = 1e-6
        report = StateComparison(
            total_params=10,
            changed_params=5,
            percent_changed=50.0,
            median_change=0.1,
            mean_change=0.1,
            max_change=0.5,
            min_change=0.0,
            change_threshold=threshold,
        )
        output = format_state_comparison(report, width=80)

        # The threshold value should appear in the output in some form
        assert "1e-06" in output or "1e-6" in output or "0.000001" in output


# ---------------------------------------------------------------------------
# TestResultDataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    def test_norms_comparison_is_frozen(self) -> None:
        report = NormsComparison(
            median_ratio=1.0,
            min_ratio=0.9,
            max_ratio=1.1,
            compared_layers=2,
            skipped_layers=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.median_ratio = 99.0  # type: ignore[misc]

    def test_state_comparison_is_frozen(self) -> None:
        report = StateComparison(
            total_params=10,
            changed_params=0,
            percent_changed=0.0,
            median_change=0.0,
            mean_change=0.0,
            max_change=0.0,
            min_change=0.0,
            change_threshold=1e-8,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.total_params = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestStateSnapshot
# ---------------------------------------------------------------------------


class TestStateSnapshot:
    def test_snapshot_is_deep_copy_mutating_model_does_not_affect_snapshot(self) -> None:
        model = _make_model(seed=0)
        snapshot = state_snapshot(model)

        # Capture the snapshot values before mutation
        snapshot_leaves_before = jax.tree_util.tree_leaves(snapshot)

        # Mutate model weights by applying a different model's state
        model_different = _make_model(seed=42)
        nnx.update(model, nnx.state(model_different))

        # Snapshot leaves should be unchanged
        snapshot_leaves_after = jax.tree_util.tree_leaves(snapshot)
        assert all(
            jnp.allclose(a, b)
            for a, b in zip(snapshot_leaves_before, snapshot_leaves_after)
        )

    def test_snapshot_tree_structure_matches_nnx_state(self) -> None:
        model = _make_model(seed=0)
        snapshot = state_snapshot(model)
        live_state = nnx.state(model)

        snapshot_leaves = jax.tree_util.tree_leaves(snapshot)
        live_leaves = jax.tree_util.tree_leaves(live_state)

        assert len(snapshot_leaves) == len(live_leaves)
        for snap_leaf, live_leaf in zip(snapshot_leaves, live_leaves):
            assert snap_leaf.shape == live_leaf.shape
