"""Regression tests pinning the committed confirmatory and discovery numbers."""

from __future__ import annotations

import numpy as np
import pytest

from glm53.bootstrap import bootstrap_contrasts, bootstrap_paired_difference
from glm53.io import GROUPS, load_raw_scores, load_roster, load_yaml
from glm53.measure import (
    address_difference,
    build_matrices,
    estimands,
    group_means,
    leave_one_out,
    matched_address_pairs,
)

POINT_TOL = 1e-9
CI_TOL = 1e-6


@pytest.fixture(scope="module")
def confirmatory():
    task = load_yaml("task.yaml")
    roster = load_roster(task)
    run = build_matrices(load_raw_scores("confirmatory", task), roster)
    return run, roster


@pytest.fixture(scope="module")
def confirmatory_bootstrap(confirmatory):
    run, _ = confirmatory
    return bootstrap_contrasts(run.matrices, reps=20000, seed=20260830)


def test_confirmatory_point_estimates(confirmatory):
    run, _ = confirmatory
    means = group_means(run.matrices, GROUPS)
    est = estimands(means)
    assert est["interaction"] == pytest.approx(-0.65024890176303, abs=POINT_TOL)
    assert est["F-U"] == pytest.approx(-0.36071546512109537, abs=POINT_TOL)
    assert est["FN-G"] == pytest.approx(0.28953343664193465, abs=POINT_TOL)
    assert est["U-G"] == pytest.approx(-0.26604876329315125, abs=POINT_TOL)
    assert means["famous_ai"] == pytest.approx(-0.6271410501464977, abs=POINT_TOL)
    assert means["famous_ai_real"] == pytest.approx(-0.7492752563067482, abs=POINT_TOL)
    assert means["famous_nonai"] == pytest.approx(0.2891566149096836, abs=POINT_TOL)
    assert means["unknown_ai"] == pytest.approx(-0.2664255850254023, abs=POINT_TOL)
    assert means["genpop"] == pytest.approx(-0.0003768217322510695, abs=POINT_TOL)


def test_confirmatory_interaction_interval(confirmatory_bootstrap):
    lower, upper = confirmatory_bootstrap["ci95"]["interaction"]
    assert lower == pytest.approx(-1.2099960877594156, abs=CI_TOL)
    assert upper == pytest.approx(-0.09734531995242601, abs=CI_TOL)


def test_confirmatory_component_intervals(confirmatory_bootstrap):
    ci = confirmatory_bootstrap["ci95"]
    assert ci["F-U"] == pytest.approx([-0.7838174554321631, 0.05207623134135718], abs=CI_TOL)
    assert ci["FN-G"] == pytest.approx([-0.10745282201631333, 0.6733968058180178], abs=CI_TOL)
    assert ci["U-G"] == pytest.approx([-0.6606871846812918, 0.12477478678476969], abs=CI_TOL)


def test_confirmatory_address_and_robustness(confirmatory):
    run, roster = confirmatory
    pairs = matched_address_pairs(roster)
    assert len(pairs) == 59
    point, ci = bootstrap_paired_difference(
        address_difference(run.matrices, pairs), reps=20000, seed=20260830 + 400
    )
    assert point == pytest.approx(0.022669779236015705, abs=POINT_TOL)
    assert ci == pytest.approx([-0.38126781962679174, 0.4255536458235963], abs=CI_TOL)
    loo = leave_one_out(run)
    assert loo["maximum_absolute_shift_pp"] == pytest.approx(0.038830210650418806, abs=POINT_TOL)
    assert loo["sign_flip_count"] == 0


def test_discovery_point_estimates():
    task = load_yaml("task.yaml")
    run = build_matrices(load_raw_scores("discovery", task), load_roster(task))
    means = group_means(run.matrices, GROUPS)
    assert estimands(means)["interaction"] == pytest.approx(-0.8314710848350346, abs=POINT_TOL)
    assert means["famous_ai"] == pytest.approx(-0.36501262491821507, abs=POINT_TOL)
    assert means["famous_ai_real"] == pytest.approx(-0.5967466838270982, abs=POINT_TOL)
    assert means["famous_nonai"] == pytest.approx(0.4555227077040741, abs=POINT_TOL)
    assert means["unknown_ai"] == pytest.approx(0.01125873137205369, abs=POINT_TOL)
    assert means["genpop"] == pytest.approx(0.0003229791593082026, abs=POINT_TOL)


def test_folded_transform_is_symmetric():
    from glm53.measure import folded

    assert np.allclose(folded(np.array([0.1, 0.5, 0.9])), [90.0, 50.0, 90.0])
