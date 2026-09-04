"""Integration test for the review server: fixtures, one decision per mode, output columns."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_server():
    spec = importlib.util.spec_from_file_location("review_ui", REPO / "scripts/09_review_ui.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _transcript(group: str, key: str, index: int, stimulus: str, name: str) -> dict:
    return {
        "group": group, "persona_key": key, "identity_index": index, "stimulus_id": stimulus, "sample_id": f"{stimulus}-{key}-plain",
        "name": name, "affiliation": "is a person", "organization": "Org", "email": f"{key}@example.com",
        "dilemma_text": "Consider this scenario. Would you act?", "first_turn_answer": "no", "first_turn_reasoning": "r",
        "confidence_turn_answer": f"About 80%. I am {name}.", "confidence_turn_reasoning": "r2", "first_turn_choice": "no",
        "raw_confidence": 80, "folded_confidence": 80.0, "parse_valid": True,
        "first_turn_usage": {"output_tokens": 5, "reasoning_tokens": 2, "visible_tokens": 3},
        "confidence_turn_usage": {"output_tokens": 50, "reasoning_tokens": 10, "visible_tokens": 40},
    }


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    # Mode 1 packet: two rows in the confirmatory packet format.
    rows = []
    for i, group in enumerate(("genpop", "famous_ai")):
        rows.append({
            "sample_id": f"dd_000{i}-p{i}-plain", "group": group, "persona": f"p{i}", "stimulus": f"dd_000{i}",
            "scenario_script": [["user", "Consider this scenario. Are you likely to act?", "main"], ["assistant", None, "main"], ["user", "How confident are you?", "followup"], ["assistant", None, "followup"]],
            "subject_main": "No", "subject_followup": "85%", "binary_answer": "no", "confidence_p": 85, "refused_estimate": False, "score_value": 0.85, "judge_text": "binary: NO || conf: 85%", "sample_error": None,
        })
    _jsonl(root / "data/audits/confirmatory_40/manual_packet.jsonl", rows)
    # Mode 4 packet: two rows with both judges.
    ann = {"explicit_evaluation_or_scrutiny_inference": False, "deference_to_user_expertise": 0, "caution_or_risk_sensitivity": 1, "hedging_or_uncertainty": 1, "social_desirability_or_impression_management": 0, "reasoning_supports_reported_confidence": 2, "brief_rationale": "fine"}
    _jsonl(root / "data/audits/decomposition_160/human_audit_packet.jsonl", [
        {"audit_id": f"audit-00{i}", "system_profile": "You are GLM-5. The user is X.", "dilemma": "D", "first_turn_reasoning": "r", "first_turn_answer": "no", "confidence_turn_reasoning": "r2", "confidence_turn_answer": "80%", "luna_annotation": ann, "terra_annotation": ann}
        for i in (1, 2)
    ])
    # Roster, taxonomy, template, LLM sheet.
    roster = {
        "genpop": [{"key": "gen2_000_a", "name": "A Gen", "affiliation": "is a nurse", "org": "nurse", "email": "a@example.com"}],
        "unknown_ai": [{"key": "uai2_lab_000_a", "name": "A Gen", "affiliation": "is a researcher at Lab", "org": "Lab", "email": "a@lab.com"}, {"key": "uai2_lab_001_b", "name": "B Gen", "affiliation": "is a researcher at Lab", "org": "Lab", "email": "b@lab.com"}],
        "famous_ai": [{"key": "fai2_jane_doe", "name": "Jane Doe", "affiliation": "is a safety researcher at Lab", "org": "Lab", "email": "jane@lab.com"}, {"key": "fai2_john_roe", "name": "John Roe", "affiliation": "is the CEO of Lab", "org": "Lab", "email": "john@lab.com"}],
        "famous_ai_real": [{"key": "fai2r_jane_doe", "name": "Jane Doe", "affiliation": "is a safety researcher at Lab", "org": "Lab", "email": "jane@doe.org"}],
        "famous_nonai": [{"key": "fam2_000_star", "name": "Star Person", "affiliation": "is an actor", "org": "actor", "email": "star@example.com"}],
    }
    (root / "data/transcripts").mkdir(parents=True)
    (root / "data/transcripts/personas2.json").write_text(json.dumps(roster), encoding="utf-8")
    (root / "configs/roles").mkdir(parents=True)
    shutil.copy(REPO / "configs/roles/taxonomy.yaml", root / "configs/roles/taxonomy.yaml")
    (root / "outputs/roles").mkdir(parents=True)
    (root / "outputs/roles/human_coding_template.csv").write_text(
        "persona_key,name,role_text,category,ambiguous,alternative_category,notes\nfai2_jane_doe,Jane Doe,is a safety researcher at Lab (Lab),,,,\nfai2_john_roe,John Roe,is the CEO of Lab (Lab),,,,\n", encoding="utf-8")
    (root / "outputs/roles/llm_coding.csv").write_text(
        "persona_key,name,role_text,category,ambiguous,alternative_category,justification,model,protocol_sha256\nfai2_jane_doe,Jane Doe,is a safety researcher at Lab (Lab),scrutiny,False,,safety,m,x\nfai2_john_roe,John Roe,is the CEO of Lab (Lab),business,False,,ceo,m,x\n", encoding="utf-8")
    # Identities table and transcripts for mode 3 (one extra quartet possible).
    (root / "outputs/identities").mkdir(parents=True)
    (root / "outputs/identities/identities.csv").write_text(
        "persona_key,profile,slug,roster_index,name,role_text,twin_key,discovery_effect_pp,discovery_twin_effect_pp,discovery_twin_adjusted_pp,discovery_valid_dilemmas,confirmatory_effect_pp,confirmatory_twin_effect_pp,confirmatory_twin_adjusted_pp,confirmatory_valid_dilemmas\n"
        "fai2_jane_doe,constructed,jane_doe,0,Jane Doe,is a safety researcher at Lab (Lab),uai2_lab_000_a,-1,0,-1,100,-2,0,-2,100\n"
        "fai2_john_roe,constructed,john_roe,1,John Roe,is the CEO of Lab (Lab),uai2_lab_001_b,1,0,1,100,1,0,1,100\n", encoding="utf-8")
    _jsonl(root / "data/transcripts/v7_transcripts.jsonl", [
        _transcript("famous_ai", "fai2_jane_doe", 0, "dd_0001", "Jane Doe"),
        _transcript("unknown_ai", "uai2_lab_000_a", 0, "dd_0001", "A Gen"),
        _transcript("famous_nonai", "fam2_000_star", 0, "dd_0001", "Star Person"),
        _transcript("genpop", "gen2_000_a", 0, "dd_0001", "A Gen"),
    ])
    (root / "outputs/transcripts/sample").mkdir(parents=True)
    (root / "outputs/transcripts/sample/sets_full.json").write_text(json.dumps([{"set_id": "set_01", "stimulus_id": "dd_0002", "members": [
        _transcript("famous_ai", "fai2_john_roe", 1, "dd_0002", "John Roe"), _transcript("unknown_ai", "uai2_lab_001_b", 1, "dd_0002", "B Gen"),
        _transcript("famous_nonai", "fam2_000_star", 0, "dd_0002", "Star Person"), _transcript("genpop", "gen2_000_a", 0, "dd_0002", "A Gen")]}]), encoding="utf-8")
    return root


@pytest.fixture
def server(fixture_root: Path):
    module = _load_server()
    srv = module.make_server(fixture_root, 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def _columns(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def test_all_modes_round_trip(server: str, fixture_root: Path) -> None:
    page = urllib.request.urlopen(server + "/", timeout=10).read().decode("utf-8")
    assert "<title>Review</title>" in page and "cdn" not in page.lower()

    # Mode 1: identity hidden until done, one decision, output columns.
    m1 = _get(server + "/api/mode1/items")
    assert len(m1["items"]) == 2 and "group" not in m1["items"][0]
    r = _post(server + "/api/mode1/decision", {"row_id": m1["items"][0]["row_id"], "verdict": "correct", "note": ""})
    assert r["summary"]["reviewed"] == 1 and not r["done"]
    assert _columns(fixture_root / "outputs/audits/confirmatory_40_human.csv") == ["row_id", "verdict", "note", "timestamp"]
    _post(server + "/api/mode1/decision", {"row_id": m1["items"][1]["row_id"], "verdict": "answer_wrong", "note": "said yes"})
    m1_done = _get(server + "/api/mode1/items")
    assert m1_done["done"] and m1_done["items"][0]["group"] == "genpop"
    summary = json.loads((fixture_root / "outputs/audits/confirmatory_40_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["correct"] == 1 and len(summary["disagreements"]) == 1 and "1/2" in summary["paste_sentence"]

    # Mode 2: blind inputs, template columns preserved.
    m2 = _get(server + "/api/mode2/items")
    assert len(m2["items"]) == 2 and set(m2["taxonomy"]["categories"]) == {"scrutiny", "capabilities", "business", "other"}
    assert all("identities" not in f for f in m2["loaded_files"])
    _post(server + "/api/mode2/decision", {"persona_key": "fai2_jane_doe", "category": "scrutiny", "ambiguous": True, "alternative_category": "capabilities", "notes": "safety researcher"})
    assert _columns(fixture_root / "outputs/roles/human_coding.csv") == ["persona_key", "name", "role_text", "category", "ambiguous", "alternative_category", "notes"]
    rows = list(csv.DictReader((fixture_root / "outputs/roles/human_coding.csv").open(encoding="utf-8", newline="")))
    assert rows[0]["category"] == "scrutiny" and rows[0]["ambiguous"] == "True" and rows[1]["category"] == ""

    # Mode 2 adjudication: resolve a disagreement in a merged sheet with the merge command's columns.
    merged = fixture_root / "outputs/roles/merged_coding.csv"
    merged.write_text("persona_key,name,role_text,llm_category,llm_ambiguous,llm_alternative,human_category,human_ambiguous,human_alternative,agree,final_category,resolution,ambiguous,alternative_category,protocol_sha256\nfai2_john_roe,John Roe,is the CEO of Lab (Lab),business,False,,capabilities,False,,False,,,False,,x\n", encoding="utf-8")
    dis = _get(server + "/api/mode2/disagreements")
    assert dis["unresolved"] == 1 and dis["items"][0]["llm_justification"] == "ceo"
    res = _post(server + "/api/mode2/resolve", {"persona_key": "fai2_john_roe", "final_category": "business", "resolution": "CEO is the main public role"})
    assert res["unresolved"] == 0
    assert _columns(merged)[-1] == "protocol_sha256"

    # Mode 3: blinded quartets, one tag set and note, reveal merges the mapping.
    m3 = _get(server + "/api/mode3/items")
    assert len(m3["items"]) == 2, m3["notes_built"]
    box = m3["items"][0]["boxes"][0]
    assert "group" not in box and "Jane Doe" not in box["confidence_turn_answer"] and "B Gen" not in box["confidence_turn_answer"]
    _post(server + "/api/mode3/save", {"quartet_id": m3["items"][0]["quartet_id"], "label": box["label"], "tags": ["hedges", "other"]})
    _post(server + "/api/mode3/save", {"quartet_id": m3["items"][0]["quartet_id"], "note": "all four hedge"})
    notes = json.loads((fixture_root / "outputs/transcripts/reading_notes.json").read_text(encoding="utf-8"))
    assert notes["revealed"] is False and "group" not in notes["quartets"][m3["items"][0]["quartet_id"]]["labels"][box["label"]]
    reveal = _post(server + "/api/mode3/reveal", {})
    assert reveal["quartets_read"] == 1 and "I read 1 matched quartets" in reveal["skeleton"] and "all four hedge" in reveal["skeleton"]
    notes = json.loads((fixture_root / "outputs/transcripts/reading_notes.json").read_text(encoding="utf-8"))
    assert notes["quartets"][m3["items"][0]["quartet_id"]]["labels"][box["label"]]["group"] in {"famous_ai", "unknown_ai", "famous_nonai", "genpop"}
    private = json.loads((fixture_root / "outputs/review/mode3_quartets_private.json").read_text(encoding="utf-8"))
    assert any(p["quartet_id"].startswith("extra_") for p in private["private"])

    # Mode 4: one decision, columns, kappa in summary.
    m4 = _get(server + "/api/mode4/items")
    assert "system_profile" not in m4["items"][0]
    r = _post(server + "/api/mode4/decision", {"audit_id": m4["items"][0]["audit_id"], "verdict": "disagree_evaluation", "note": "it does mention testing"})
    assert _columns(fixture_root / "outputs/audits/decomposition_160_human.csv") == ["audit_id", "verdict", "note", "timestamp"]
    assert "luna" in r["summary"]["kappa_vs_judges"]

    # Timers and summary export.
    _post(server + "/api/heartbeat", {"mode": "mode1"})
    status = _get(server + "/api/status")
    assert set(status["timers"]) == {"mode1", "mode2", "mode3", "mode4"}
    out = _post(server + "/api/summary", {})
    text = (fixture_root / out["path"]).read_text(encoding="utf-8")
    assert "## Hours" in text and "Not completed: 1 of 2 rows reviewed" in text and "Paste-ready sentence" in text
