"""Stage 5: confidence-turn response length, judge annotations, and matched reading samples.

Subcommands:
  analyze  length by group and four-group interaction; judge interactions and kappas (default)
  sample   seeded matched dilemma transcript sets for human reading, with a blinded copy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, finish, print_plan, provenance, rel, stage_parser  # noqa: E402

from glm53.io import load_yaml, read_json, write_csv, write_json  # noqa: E402
from glm53.transcripts import (  # noqa: E402
    annotation_analysis,
    blinded_copy,
    length_analysis,
    load_judgments,
    load_transcripts,
    matched_sets,
    sets_frame,
)

STAGE = "transcripts"
TRANSCRIPTS = REPO_ROOT / "data/transcripts/v7_transcripts.jsonl"
JUDGMENTS = REPO_ROOT / "data/judgments/v7_content"
COMMITTED = REPO_ROOT / "data/judgments/reports"


def compare(observed: float, pinned: float, tol: float) -> dict[str, object]:
    diff = float(observed - pinned)
    return {"observed": observed, "pinned": pinned, "diff": diff, "within_tolerance": abs(diff) <= tol}


def cmd_analyze(args, analysis) -> int:
    inputs = [TRANSCRIPTS, JUDGMENTS / "luna_max/rows", JUDGMENTS / "terra_high/rows"]
    if args.dry_run:
        print_plan(STAGE, inputs, [args.output_root / "summary.json", args.output_root / "length_by_group.csv"], {"api_calls": 0})
        return 0
    if not TRANSCRIPTS.exists():
        print(f"missing {rel(TRANSCRIPTS)}; see data/transcripts/POINTERS.md")
        return 2
    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seed_det = int(analysis["bootstrap"]["seeds"]["decomposition"])
    rows = load_transcripts(TRANSCRIPTS)
    lengths = length_analysis(rows, reps=reps, seed=seed_det)
    judged = load_judgments(JUDGMENTS)
    annotations = annotation_analysis(rows, judged, reps=reps, seed=seed_det + 1000)
    tol_point = float(analysis["tolerances"]["point_estimate"])
    tol_ci = float(analysis["tolerances"]["ci_endpoint"])
    checks = {}
    det_path, ann_path = COMMITTED / "deterministic_analysis.json", COMMITTED / "annotation_analysis.json"
    if det_path.exists():
        committed = read_json(det_path)["reasoning_and_response_length_interactions"]
        for key in ("confidence_turn_visible_tokens", "confidence_visible_sentence_count"):
            checks[f"{key}_point"] = compare(lengths[key]["interaction"], committed[key]["interaction"], tol_point)
            checks[f"{key}_ci_lower"] = compare(lengths[key]["ci95"][0], committed[key]["ci95"][0], tol_ci)
            checks[f"{key}_ci_upper"] = compare(lengths[key]["ci95"][1], committed[key]["ci95"][1], tol_ci)
    if ann_path.exists():
        committed = read_json(ann_path)
        for field in ("explicit_evaluation_or_scrutiny_inference", "deference_to_user_expertise"):
            checks[f"{field}_kappa"] = compare(annotations["agreement"][field]["kappa"], committed["agreement"][field]["kappa"], tol_point)
            checks[f"{field}_point"] = compare(annotations["dimensions"][field]["combined"]["interaction"], committed["dimensions"][field]["combined"]["interaction"], tol_point)
            for side, label in ((0, "lower"), (1, "upper")):
                checks[f"{field}_ci_{label}"] = compare(annotations["dimensions"][field]["combined"]["ci95"][side], committed["dimensions"][field]["combined"]["ci95"][side], tol_ci)
    failures = [k for k, v in checks.items() if not v["within_tolerance"]]
    table = []
    for key in ("confidence_turn_visible_tokens", "confidence_visible_sentence_count", "confidence_turn_reasoning_tokens", "first_turn_visible_tokens"):
        for group, value in lengths[key]["raw_group_means"].items():
            table.append({"measure": key, "unit": lengths[key]["unit"], "group": group, "mean": value})
    import pandas as pd

    write_csv(args.output_root / "length_by_group.csv", pd.DataFrame(table))
    summary = {
        "units": {
            "tokens": "provider-reported output tokens of the confidence turn minus its reasoning tokens",
            "sentences": "count of sentence-ending punctuation splits in the visible confidence-turn text",
            "evaluation_language": "difference in the fraction of transcripts with explicit evaluation language (0 to 1), judges averaged",
            "deference": "difference in mean rating on a 0 to 2 scale, judges averaged",
            "interaction": "(famous AI - unknown twin) - (famous non-AI - general population), identities weighted equally, centered on the per-dilemma general-population mean",
        },
        "length": lengths,
        "annotations": annotations,
        "dropped_dimensions": {
            f: {"kappa": annotations["agreement"][f]["kappa"], "reason": "inter-judge agreement below the frozen threshold"}
            for f in annotations["agreement"]
            if not annotations["agreement"][f]["adequate"]
        },
        "checks": checks,
        "all_checks_pass": not failures,
        "failures": failures,
        "reps": reps,
        "provenance": provenance(inputs[:1] + [det_path, ann_path], [REPO_ROOT / "configs/analysis.yaml"]),
    }
    finish(STAGE, args.output_root, summary)
    print(f"checks: {len(checks)}, failures: {failures}")
    for key in ("confidence_turn_visible_tokens", "confidence_visible_sentence_count"):
        print(key, round(lengths[key]["interaction"], 3), [round(x, 3) for x in lengths[key]["ci95"]])
    for field in ("explicit_evaluation_or_scrutiny_inference", "deference_to_user_expertise"):
        d = annotations["dimensions"][field]["combined"]
        print(field, round(d["interaction"], 4), [round(x, 4) for x in d["ci95"]], "kappa", round(annotations["agreement"][field]["kappa"], 3))
    return 0 if not failures else 1


def cmd_sample(args, analysis) -> int:
    out = args.output_root / "sample"
    if args.dry_run:
        print_plan("transcripts/sample", [TRANSCRIPTS], [out / "sets_full.json", out / "sets_blinded.json", out / "sets.csv", out / "reading_blinded.md"], {"api_calls": 0, "sets": args.sets})
        return 0
    seed = args.seed or int(analysis["bootstrap"]["seeds"]["transcripts"])
    rows = load_transcripts(TRANSCRIPTS)
    sets = matched_sets(rows, n_sets=args.sets, seed=seed)
    rng = np.random.default_rng(seed + 1)
    blinded = [blinded_copy(item, rng) for item in sets]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "sets_full.json", sets)
    write_json(out / "sets_blinded.json", blinded)
    write_csv(out / "sets.csv", sets_frame(sets))
    lines = ["# Blinded matched transcript sets", "", f"seed {seed}; profile fields removed; identity strings replaced by placeholders.", ""]
    for item in blinded:
        lines.append(f"## {item['set_id']} ({item['stimulus_id']})")
        lines.append("")
        lines.append(item["members"][0]["dilemma_text"])
        lines.append("")
        for m in item["members"]:
            lines += [f"### {m['reader_label']}", "", "First-turn reasoning:", "", m["first_turn_reasoning"], "", f"First answer: {m['first_turn_answer']}", "", "Confidence-turn reasoning:", "", m["confidence_turn_reasoning"], "", "Confidence answer:", "", m["confidence_turn_answer"], ""]
    (out / "reading_blinded.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(sets)} matched sets to {rel(out)} (key in sets_full.json; blinded copy in sets_blinded.json)")
    return 0


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("command", nargs="?", default="analyze", choices=("analyze", "sample"))
    parser.add_argument("--reps", type=int)
    parser.add_argument("--sets", type=int, default=6)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    analysis = load_yaml("analysis.yaml")
    return {"analyze": cmd_analyze, "sample": cmd_sample}[args.command](args, analysis)


if __name__ == "__main__":
    raise SystemExit(main())
