"""Pins the committed decomposition points and the judge-agreement statistics."""

from __future__ import annotations

import numpy as np
import pytest

from glm53 import REPO_ROOT
from glm53.decompose import choice_standardized, interaction, matched_same_choice_point, outcome_matrices
from glm53.io import load_raw_scores, load_roster, load_yaml
from glm53.transcripts import load_judgments, sentence_count

POINT_TOL = 1e-9


@pytest.fixture(scope="module")
def outcomes():
    task = load_yaml("task.yaml")
    return outcome_matrices(load_raw_scores("confirmatory", task), load_roster(task))


def test_choice_and_confidence_points(outcomes):
    choice, conf = outcomes
    assert interaction(choice)[0] * 100.0 == pytest.approx(-0.08155545908178101, abs=POINT_TOL)
    assert interaction(conf)[0] == pytest.approx(-0.6502489017630302, abs=POINT_TOL)
    assert choice_standardized(conf, choice) == pytest.approx(-0.6594210555395004, abs=POINT_TOL)


def test_matched_same_choice(outcomes):
    choice, conf = outcomes
    matched = matched_same_choice_point(conf, choice)
    assert matched["interaction_pp"] == pytest.approx(-0.7354605912535765, abs=POINT_TOL)
    assert matched["famous_unknown_retained_cells"] == 6279
    assert matched["famous_nonai_genpop_retained_cells"] == 6336


def test_judge_rows_complete():
    judged = load_judgments(REPO_ROOT / "data/judgments/v7_content")
    assert len(judged) == 2000
    assert all(set(v) == {"luna_max", "terra_high"} for v in judged.values())


def test_sentence_count():
    assert sentence_count("") == 0
    assert sentence_count("One. Two? Three!") == 3
    assert np.isfinite(sentence_count("No terminal punctuation"))
