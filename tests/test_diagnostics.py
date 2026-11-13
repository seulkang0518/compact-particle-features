import json

import pytest
import torch

from compact_features.diagnostics import diagnose_features


def test_identical_geometry_and_different_channel_counts():
    torch.manual_seed(8)
    teacher = {"layer2/spatial": torch.randn(3, 11, 4)}
    # Isometric embedding, then positive rescaling: same normalized geometry.
    student = {key: torch.cat([value, torch.zeros(3, 11, 3)], -1) * 2
               for key, value in teacher.items()}
    report = diagnose_features(student, teacher, 4, [.05, .2])
    metrics = report["mean"]
    assert metrics["normalized_distance_rmse"] < 1e-12
    assert metrics["neighbor_overlap_at_k"] == 1.
    assert metrics["student_over_teacher_within_group_variance_fraction"] == pytest.approx(1.)
    assert metrics["student_over_teacher_generated_spread_normalized"] == pytest.approx(1.)
    for tau in [.05, .2]:
        assert metrics[f"student_negative_effective_neighbors_tau_{tau:g}"] <= 3. + 1e-12
        assert metrics[f"student_positive_effective_neighbors_tau_{tau:g}"] <= 7. + 1e-12
    json.dumps(report, allow_nan=False)


def test_collapsed_student_detected_and_report_is_finite():
    torch.manual_seed(3)
    teacher = {"conv1/mean": torch.randn(2, 9, 5)}
    student = {"conv1/mean": torch.zeros(2, 9, 2)}
    metrics = diagnose_features(student, teacher, 3, [.1])["mean"]
    assert metrics["student_collapsed_position_fraction"] == 1.
    assert metrics["student_collapsed_generated_position_fraction"] == 1.
    assert metrics["normalized_distance_rmse"] > 0.
    assert metrics["student_positive_effective_neighbors_tau_0.1"] == pytest.approx(6.)
    assert metrics["student_negative_effective_neighbors_tau_0.1"] == pytest.approx(2.)
    json.dumps(metrics, allow_nan=False)


def test_query_self_is_excluded_and_selectivity_changes_with_temperature():
    x = torch.tensor([[[0.], [1.], [3.], [.05], [10.], [20.], [30.], [40.]]])
    metrics = diagnose_features({"a": x}, {"a": x}, 3, [1e-4, 1e4])["mean"]
    # Including the zero-distance generated self would give ~1 at every low tau.
    # This generated configuration still gives correctly bounded Q-1 entropy.
    assert metrics["teacher_negative_effective_neighbors_tau_10000"] == pytest.approx(2., rel=1e-6)
    assert metrics["teacher_positive_effective_neighbors_tau_10000"] == pytest.approx(5., rel=1e-6)
    assert metrics["teacher_positive_effective_neighbors_tau_0.0001"] < 1.01
    assert metrics["neighbor_k"] == 5.


def test_branches_are_averaged_equally_and_inputs_remain_differentiable():
    x = torch.randn(2, 9, 4, requires_grad=True)
    teacher = {"a": x, "b": x.detach() + .4}
    student = {"a": x, "b": torch.zeros_like(x)}
    report = diagnose_features(student, teacher, 3, [.1])
    expected = sum(v["normalized_distance_rmse"] for v in report["branches"].values()) / 2
    assert report["mean"]["normalized_distance_rmse"] == pytest.approx(expected)
    assert x.requires_grad and x.grad is None


@pytest.mark.parametrize("queries,taus", [(1,[.1]), (3,[]), (3,[0.]), (3,[float('nan')]), (3,[.1,.1])])
def test_invalid_query_counts_or_temperatures_rejected(queries, taus):
    values={"a": torch.randn(1, 8, 2)}
    with pytest.raises(ValueError):
        diagnose_features(values, values, queries, taus)


def test_incompatible_and_nonfinite_branches_rejected():
    values={"a": torch.randn(1, 8, 2)}
    with pytest.raises(ValueError, match="branch names"):
        diagnose_features(values, {"b": values["a"]}, 3, [.1])
    with pytest.raises(ValueError, match="mismatched"):
        diagnose_features(values, {"a": torch.randn(2, 8, 4)}, 3, [.1])
    bad=values["a"].clone(); bad[0,0,0]=float('inf')
    with pytest.raises(ValueError, match="Nonfinite"):
        diagnose_features({"a": bad}, values, 3, [.1])
