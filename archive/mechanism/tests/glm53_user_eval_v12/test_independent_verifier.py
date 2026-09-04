from __future__ import annotations

import json
from pathlib import Path

from src.glm53_user_eval.v12.fact_validation import (
    analyze_primary,
    analyze_verifier,
    atomic_json,
    build_verifier_schedule,
)
from src.glm53_user_eval.v12.independent_verifier import verify_v12

from .conftest import judgment


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_import_isolation() -> None:
    source = Path(
        "src/glm53_user_eval/v12/independent_verifier.py"
    ).read_text(encoding="utf-8")
    assert "fact_validation" not in source


def test_independent_verifier_recomputes_full_result(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(dataset, frozen_rows)
    primary_root = tmp_path / "primary"
    primary_rows = [judgment(row) for row in frozen_rows]
    for record in primary_rows:
        atomic_json(primary_root / "rows" / f"{record['sample_id']}.json", record)
    primary = analyze_primary(frozen_rows, primary_rows, output_root=primary_root)
    primary_path = tmp_path / "primary_analysis.json"
    atomic_json(primary_path, primary)

    schedule = build_verifier_schedule(frozen_rows, primary)
    schedule_path = tmp_path / "verifier_schedule.json"
    atomic_json(schedule_path, schedule)
    by_id = {row["sample_id"]: row for row in frozen_rows}
    verifier_root = tmp_path / "verifier"
    verifier_rows = [
        judgment(by_id[sample_id], pass_kind="verifier")
        for sample_id in schedule["sample_ids"]
    ]
    for record in verifier_rows:
        atomic_json(verifier_root / "rows" / f"{record['sample_id']}.json", record)
    verifier = analyze_verifier(
        frozen_rows,
        primary_rows,
        verifier_rows,
        schedule,
        output_root=verifier_root,
    )
    verifier_path = tmp_path / "verifier_analysis.json"
    atomic_json(verifier_path, verifier)

    report = verify_v12(
        dataset_path=dataset,
        primary_root=primary_root,
        primary_analysis_path=primary_path,
        verifier_root=verifier_root,
        verifier_schedule_path=schedule_path,
        verifier_analysis_path=verifier_path,
    )
    assert report["passed"] is True
    assert report["scientific_gate_passed"] is True
    assert all(report["comparisons"].values())


def test_independent_verifier_rejects_prompt_label_leakage(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(dataset, frozen_rows)
    primary_root = tmp_path / "primary"
    primary_rows = [judgment(row) for row in frozen_rows]
    for record in primary_rows:
        atomic_json(primary_root / "rows" / f"{record['sample_id']}.json", record)
    primary = analyze_primary(frozen_rows, primary_rows, output_root=primary_root)
    primary_path = tmp_path / "primary_analysis.json"
    atomic_json(primary_path, primary)
    schedule = build_verifier_schedule(frozen_rows, primary)
    schedule_path = tmp_path / "verifier_schedule.json"
    atomic_json(schedule_path, schedule)
    by_id = {row["sample_id"]: row for row in frozen_rows}
    verifier_root = tmp_path / "verifier"
    verifier_rows = [
        judgment(by_id[sample_id], pass_kind="verifier")
        for sample_id in schedule["sample_ids"]
    ]
    for record in verifier_rows:
        atomic_json(verifier_root / "rows" / f"{record['sample_id']}.json", record)
    verifier = analyze_verifier(
        frozen_rows,
        primary_rows,
        verifier_rows,
        schedule,
        output_root=verifier_root,
    )
    verifier_path = tmp_path / "verifier_analysis.json"
    atomic_json(verifier_path, verifier)

    contaminated = primary_root / "rows" / f"{frozen_rows[0]['sample_id']}.json"
    record = json.loads(contaminated.read_text(encoding="utf-8"))
    record["request"]["messages"][0]["content"] += " eval"
    atomic_json(contaminated, record)
    report = verify_v12(
        dataset_path=dataset,
        primary_root=primary_root,
        primary_analysis_path=primary_path,
        verifier_root=verifier_root,
        verifier_schedule_path=schedule_path,
        verifier_analysis_path=verifier_path,
    )
    assert report["passed"] is False
    assert report["comparisons"]["prompt_blinding"] is False


def test_independent_verifier_rejects_tampered_primary_pass(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    dataset = tmp_path / "samples.jsonl"
    _write_jsonl(dataset, frozen_rows)
    primary_root = tmp_path / "primary"
    primary_rows = [judgment(row) for row in frozen_rows]
    for record in primary_rows:
        atomic_json(primary_root / "rows" / f"{record['sample_id']}.json", record)
    primary = analyze_primary(frozen_rows, primary_rows, output_root=primary_root)
    primary["passed"] = False
    primary_path = tmp_path / "primary_analysis.json"
    atomic_json(primary_path, primary)
    schedule = build_verifier_schedule(frozen_rows, primary)
    schedule_path = tmp_path / "verifier_schedule.json"
    atomic_json(schedule_path, schedule)
    by_id = {row["sample_id"]: row for row in frozen_rows}
    verifier_root = tmp_path / "verifier"
    verifier_rows = [
        judgment(by_id[sample_id], pass_kind="verifier")
        for sample_id in schedule["sample_ids"]
    ]
    for record in verifier_rows:
        atomic_json(verifier_root / "rows" / f"{record['sample_id']}.json", record)
    verifier = analyze_verifier(
        frozen_rows,
        primary_rows,
        verifier_rows,
        schedule,
        output_root=verifier_root,
    )
    verifier_path = tmp_path / "verifier_analysis.json"
    atomic_json(verifier_path, verifier)
    report = verify_v12(
        dataset_path=dataset,
        primary_root=primary_root,
        primary_analysis_path=primary_path,
        verifier_root=verifier_root,
        verifier_schedule_path=schedule_path,
        verifier_analysis_path=verifier_path,
    )
    assert report["passed"] is False
    assert report["comparisons"]["primary_passed"] is False
