"""The committed final role labels are the ones the headline role contrast used."""

from __future__ import annotations

import csv
import json

from glm53 import REPO_ROOT
from glm53.roles import CATEGORIES, label_hash

REVIEW = REPO_ROOT / "data/review"


def test_merged_labels_match_contrast_hash() -> None:
    with (REVIEW / "roles/merged_coding.csv").open(encoding="utf-8", newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert len(merged) == 70
    labels = {r["persona_key"]: r["final_category"] for r in merged}
    assert set(labels.values()) <= set(CATEGORIES)
    contrast = json.loads((REVIEW / "identities/role_contrast.json").read_text(encoding="utf-8"))
    assert label_hash(labels) == contrast["labels_sha256"]
    disagreements = [r for r in merged if r["agree"] != "True"]
    assert len(disagreements) == 16 and all(r["final_category"] and r["resolution"] for r in disagreements)


def test_human_sheet_is_complete_and_template_shaped() -> None:
    with (REVIEW / "roles/human_coding.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["persona_key", "name", "role_text", "category", "ambiguous", "alternative_category", "notes"]
    assert len(rows) == 70 and all(r["category"] in CATEGORIES for r in rows)
