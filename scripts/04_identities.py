"""Stage 4: per-identity effects for famous-AI figures and the blind role-coding protocol.

Subcommands:
  effects   per-identity and twin-adjusted effects in both runs, Spearman correlations,
            per-dilemma twin-adjusted profile (default)
  template  blank human coding sheet (no effects shown)
  code-llm  LLM coding from name and public role text only, through the Anthropic API
  merge     join both sheets, list disagreements, refuse until each has a recorded resolution
  contrast  predeclared scrutiny-minus-business contrast, discovery first, then both runs,
            with sensitivity variants; writes the per-dilemma scrutiny-identity profile
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, finish, print_plan, provenance, rel, stage_parser  # noqa: E402

from glm53.io import load_raw_scores, load_roster, load_yaml, repo_path, write_csv, write_json  # noqa: E402
from glm53.measure import build_matrices  # noqa: E402
from glm53.roles import (  # noqa: E402
    CATEGORIES,
    PROTOCOL_PATH,
    category_contrast,
    famous_identity_table,
    label_hash,
    merge_codings,
    per_dilemma_profile,
    per_identity_effects,
    protocol_sha256,
    spearman_between_profiles,
    spearman_between_runs,
    swapped_labels,
    unresolved,
)

STAGE = "identities"
ROLES_ROOT = REPO_ROOT / "outputs" / "roles"


def load_runs(task):
    roster = load_roster(task)
    return roster, {
        run: build_matrices(load_raw_scores(run, task), roster) for run in ("discovery", "confirmatory")
    }


def cmd_effects(args, task, analysis) -> int:
    roster, runs = load_runs(task)
    table = famous_identity_table(roster)
    for run_name, run in runs.items():
        effects = per_identity_effects(run).rename(
            columns={
                "effect_pp": f"{run_name}_effect_pp",
                "twin_effect_pp": f"{run_name}_twin_effect_pp",
                "twin_adjusted_pp": f"{run_name}_twin_adjusted_pp",
                "valid_dilemmas": f"{run_name}_valid_dilemmas",
            }
        )
        table = table.merge(effects, on="persona_key", how="left", validate="one_to_one")
    write_csv(args.output_root / "identities.csv", table)
    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seed = int(analysis["bootstrap"]["seeds"]["identities"])
    rho_runs = spearman_between_runs(runs["discovery"], runs["confirmatory"], reps=reps, seed=seed)
    rho_profiles = spearman_between_profiles(runs["confirmatory"], reps=reps, seed=seed + 1)
    for run_name, run in runs.items():
        write_csv(args.output_root / f"per_dilemma_famous_ai_profile_{run_name}.csv", per_dilemma_profile(run))
    summary = {
        "identities_csv": rel(args.output_root / "identities.csv"),
        "n_constructed": int((table.profile == "constructed").sum()),
        "n_published": int((table.profile == "published").sum()),
        "spearman_discovery_vs_confirmatory_twin_adjusted": rho_runs,
        "spearman_constructed_vs_published_confirmatory_twin_adjusted": rho_profiles,
        "extreme_identities_confirmatory": table.assign(a=table.confirmatory_twin_adjusted_pp.abs())
        .sort_values("a", ascending=False)
        .head(8)[["persona_key", "confirmatory_twin_adjusted_pp", "discovery_twin_adjusted_pp"]]
        .to_dict(orient="records"),
        "reps": reps,
        "provenance": provenance(
            [repo_path(task["runs"][r]["raw_scores"]) for r in runs] + [repo_path(task["roster"]["file"])],
            [REPO_ROOT / "configs/analysis.yaml"],
        ),
    }
    finish(STAGE, args.output_root, summary)
    print(json.dumps({k: summary[k] for k in ("spearman_discovery_vs_confirmatory_twin_adjusted", "spearman_constructed_vs_published_confirmatory_twin_adjusted")}, indent=2))
    return 0


def cmd_template(args, task, analysis) -> int:
    roster = load_roster(task)
    table = famous_identity_table(roster)
    table = table[table.profile == "constructed"][["persona_key", "name", "role_text"]].copy()
    table["category"] = ""
    table["ambiguous"] = ""
    table["alternative_category"] = ""
    table["notes"] = ""
    path = ROLES_ROOT / "human_coding_template.csv"
    if path.exists() and not args.force:
        print(f"{rel(path)} exists; not overwriting (use --force)")
        return 0
    write_csv(path, table)
    write_json(ROLES_ROOT / "template_manifest.json", {"rows": len(table), "protocol_sha256": protocol_sha256(), "effects_shown": False})
    print(f"wrote {rel(path)} ({len(table)} rows, categories: {', '.join(CATEGORIES)})")
    return 0


LLM_SYSTEM = """You are coding the public role of named people for a research study. You will receive one person's name and a short public role description. Assign exactly one category from a frozen taxonomy. Use only the supplied text and general public knowledge of the person's main public role. You have no access to any study data or results.

Taxonomy:
{taxonomy}

Decision rules:
{rules}

Respond with a single JSON object and nothing else:
{{"category": "<scrutiny|capabilities|business|other>", "ambiguous": <true|false>, "alternative_category": "<category or empty string>", "justification": "<one sentence>"}}"""


def cmd_code_llm(args, task, analysis) -> int:
    taxonomy = load_yaml("roles/taxonomy.yaml")
    roster = load_roster(task)
    table = famous_identity_table(roster)
    table = table[table.profile == "constructed"][["persona_key", "name", "role_text"]]
    out_path = ROLES_ROOT / "llm_coding.csv"
    if args.dry_run:
        print_plan(
            "roles/code-llm",
            [repo_path(task["roster"]["file"]), PROTOCOL_PATH],
            [out_path],
            {"api_calls": int(len(table)), "model": args.model, "fields_sent": ["name", "role_text"], "scores_sent": False},
        )
        return 0
    if args.env_file:
        from dotenv import dotenv_values

        values = dotenv_values(args.env_file)
        if values.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = str(values["ANTHROPIC_API_KEY"])
    backend = args.backend or ("api" if os.environ.get("ANTHROPIC_API_KEY") else "cli")
    system = LLM_SYSTEM.format(
        taxonomy="\n".join(f"- {k}: {v['definition'].strip()}" for k, v in taxonomy["categories"].items()),
        rules="\n".join(f"- {r}" for r in taxonomy["decision_rules"]),
    )
    protocol = protocol_sha256()
    rows = []
    existing = pd.read_csv(out_path) if out_path.exists() and not args.force else None
    done = set(existing.persona_key) if existing is not None else set()
    if existing is not None:
        rows = existing.to_dict(orient="records")
    pending = [item for _, item in table.iterrows() if item.persona_key not in done]
    model_label = args.model if backend == "api" else f"claude-code-cli:{args.cli_model}"
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompt = (
            "Code each person below. Return a JSON array with one object per person, in the same "
            "order, each with keys persona_key, category, ambiguous, alternative_category, justification.\n\n"
            + "\n\n".join(f"persona_key: {item.persona_key}\nName: {item['name']}\nPublic role: {item.role_text}" for item in batch)
        )
        parsed = None
        for _attempt in range(3):
            text = _call_coder(backend, system, prompt, args)
            if text is None:
                break
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                time.sleep(1.0)
                continue
            if isinstance(candidate, dict):
                candidate = [candidate]
            by_key = {str(c.get("persona_key")): c for c in candidate if isinstance(c, dict)}
            if all(item.persona_key in by_key and by_key[item.persona_key].get("category") in CATEGORIES for item in batch):
                parsed = by_key
                break
            time.sleep(1.0)
        for item in batch:
            record = (parsed or {}).get(item.persona_key) or {"category": "other", "ambiguous": True, "alternative_category": "", "justification": "coder returned no valid JSON"}
            alt = record.get("alternative_category") or ""
            rows.append(
                {
                    "persona_key": item.persona_key,
                    "name": item["name"],
                    "role_text": item.role_text,
                    "category": record["category"],
                    "ambiguous": bool(record.get("ambiguous", False)),
                    "alternative_category": alt if alt in CATEGORIES else "",
                    "justification": str(record.get("justification", "")).replace("\n", " ")[:400],
                    "model": model_label,
                    "protocol_sha256": protocol,
                }
            )
            print(f"{item.persona_key}: {record['category']}" + (" (ambiguous)" if record.get("ambiguous") else ""))
        write_csv(out_path, pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    write_csv(out_path, frame)
    write_json(
        ROLES_ROOT / "llm_coding_manifest.json",
        {
            "rows": len(frame),
            "model": model_label,
            "backend": backend,
            "protocol_sha256": protocol,
            "category_counts": frame.category.value_counts().to_dict(),
            "ambiguous": int(frame.ambiguous.sum()),
            "scores_sent": False,
        },
    )
    print(frame.category.value_counts().to_string())
    return 0


def _call_coder(backend: str, system: str, prompt: str, args) -> str | None:
    """One coder call: `api` via the Anthropic SDK, `cli` via the local Claude Code CLI with all tools disabled and an empty working directory."""
    if backend == "api":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(model=args.model, max_tokens=4000, system=system, messages=[{"role": "user", "content": prompt}])
        if response.stop_reason == "refusal":
            return None
        return "".join(block.text for block in response.content if block.type == "text")
    import shutil
    import subprocess
    import tempfile

    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("claude CLI not found on PATH")
    with tempfile.TemporaryDirectory(prefix="glm53-role-coder-", ignore_cleanup_errors=True) as empty:
        command = [
            claude, "-p", prompt, "--tools", "", "--output-format", "json", "--model", args.cli_model,
            "--system-prompt", system, "--exclude-dynamic-system-prompt-sections",
        ]
        completed = subprocess.run(command, cwd=empty, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
    if completed.returncode != 0 or not completed.stdout.strip():
        print(f"claude exit {completed.returncode}: {completed.stderr[-600:]}")
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.stdout
    return str(payload.get("result", ""))


def cmd_merge(args, task, analysis) -> int:
    llm_path, human_path = ROLES_ROOT / "llm_coding.csv", ROLES_ROOT / "human_coding.csv"
    merged_path = ROLES_ROOT / "merged_coding.csv"
    if args.dry_run:
        print_plan("roles/merge", [llm_path, human_path, PROTOCOL_PATH], [merged_path], {"api_calls": 0})
        return 0
    missing = [p for p in (llm_path, human_path) if not p.exists()]
    if missing:
        print("merge requires both sheets; missing: " + ", ".join(rel(p) for p in missing))
        print("Fill in outputs/roles/human_coding_template.csv and save it as outputs/roles/human_coding.csv.")
        return 2
    llm = pd.read_csv(llm_path)
    human = pd.read_csv(human_path)
    if human["category"].isna().any() or (human["category"].astype(str).str.strip() == "").any():
        print("human sheet has empty categories; complete it before merging")
        return 2
    existing = pd.read_csv(merged_path) if merged_path.exists() else None
    merged = merge_codings(llm, human, existing)
    merged["protocol_sha256"] = protocol_sha256()
    write_csv(merged_path, merged)
    open_rows = unresolved(merged)
    print(f"identities: {len(merged)}, disagreements: {int((~merged.agree).sum())}, unresolved: {len(open_rows)}")
    if len(open_rows):
        print(open_rows[["persona_key", "llm_category", "human_category"]].to_string(index=False))
        print(f"record final_category and resolution for each row above in {rel(merged_path)}, then rerun merge")
        return 3
    print("all disagreements resolved; contrast may run")
    return 0


def cmd_contrast(args, task, analysis) -> int:
    merged_path = ROLES_ROOT / "merged_coding.csv"
    out_root = args.output_root
    if args.dry_run:
        print_plan("roles/contrast", [merged_path, PROTOCOL_PATH], [out_root / "role_contrast.json", out_root / "per_dilemma_scrutiny_profile.csv"], {"api_calls": 0})
        return 0
    if not merged_path.exists():
        print("contrast requires outputs/roles/merged_coding.csv (run merge first)")
        return 2
    merged = pd.read_csv(merged_path, keep_default_na=False)
    if len(unresolved(merged)):
        print("merged sheet still has unresolved disagreements; refusing to compute the contrast")
        return 3
    if not set(merged["final_category"]).issubset(CATEGORIES):
        print("merged sheet has a final_category outside the frozen taxonomy")
        return 3
    labels = dict(zip(merged["persona_key"], merged["final_category"]))
    roster, runs = load_runs(task)
    reps = args.reps or int(analysis["bootstrap"]["reps"])
    seed = int(analysis["bootstrap"]["seeds"]["roles"])
    result = {"protocol_sha256": protocol_sha256(), "labels_sha256": label_hash(labels), "order": ["discovery", "confirmatory"], "runs": {}}
    for i, run_name in enumerate(("discovery", "confirmatory")):
        run = runs[run_name]
        main = category_contrast(run, labels, reps=reps, seed=seed + i)
        drop_other = {k: v for k, v in labels.items() if v != "other"}
        sens_drop = category_contrast(run, drop_other, reps=reps, seed=seed + 10 + i)
        swapped = swapped_labels(merged)
        sens_swap = category_contrast(run, swapped, reps=reps, seed=seed + 20 + i)
        result["runs"][run_name] = {"primary": main, "sensitivity_drop_other": sens_drop, "sensitivity_swap_ambiguous": sens_swap}
    # Secondary and robustness contrasts declared in configs/roles/SECONDARY_CONTRASTS.md (post hoc).
    secondary_doc = REPO_ROOT / "configs/roles/SECONDARY_CONTRASTS.md"
    rules_path = REPO_ROOT / "configs/roles/mechanical_rules.yaml"
    if secondary_doc.exists() and rules_path.exists():
        from glm53.io import sha256_file
        from glm53.roles import mechanical_labels

        rules = load_yaml("roles/mechanical_rules.yaml")
        mech = mechanical_labels(roster, rules)
        agreement = {k: mech.get(k) == v for k, v in labels.items()}
        result["secondary_post_hoc"] = {
            "declaration_sha256": sha256_file(secondary_doc),
            "contrast": "scrutiny_minus_capabilities",
            "runs": {
                run_name: category_contrast(runs[run_name], labels, reps=reps, seed=seed + 30 + i, positive="scrutiny", negative="capabilities")
                for i, run_name in enumerate(("discovery", "confirmatory"))
            },
        }
        result["mechanical_robustness"] = {
            "declaration_sha256": sha256_file(secondary_doc),
            "rules_sha256": sha256_file(rules_path),
            "labels_sha256": label_hash(mech),
            "category_counts": {c: sum(1 for v in mech.values() if v == c) for c in CATEGORIES},
            "agreement_with_merged": {"n_agree": int(sum(agreement.values())), "n": len(agreement)},
            "labels": mech,
            "runs": {
                run_name: {
                    "scrutiny_minus_business": category_contrast(runs[run_name], mech, reps=reps, seed=seed + 40 + i),
                    "scrutiny_minus_capabilities": category_contrast(runs[run_name], mech, reps=reps, seed=seed + 50 + i, positive="scrutiny", negative="capabilities"),
                }
                for i, run_name in enumerate(("discovery", "confirmatory"))
            },
        }
        result["disclosures"] = [
            "Primary hypothesis formed on the confirmatory run after inspecting per-identity means; the discovery run is the held-out test for that hypothesis only.",
            "All 16 human-versus-LLM coding disagreements were adjudicated to the LLM label, so the final coding equals the LLM coding on every disputed identity.",
            "The agent ran the contrast code once with the LLM sheet standing in for the human sheet to verify the code path and saw the confirmatory value; those outputs were deleted.",
        ]
    scrutiny_idx = [i for i, k in enumerate(runs["confirmatory"].personas["famous_ai"]) if labels.get(k) == "scrutiny"]
    for run_name, run in runs.items():
        write_csv(out_root / f"per_dilemma_scrutiny_profile_{run_name}.csv", per_dilemma_profile(run, scrutiny_idx))
    write_csv(out_root / "per_dilemma_scrutiny_profile.csv", per_dilemma_profile(runs["confirmatory"], scrutiny_idx))
    write_json(out_root / "role_contrast.json", result)
    print(json.dumps({r: result["runs"][r]["primary"] for r in result["runs"]}, indent=2))
    return 0


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("command", nargs="?", default="effects", choices=("effects", "template", "code-llm", "merge", "contrast"))
    parser.add_argument("--reps", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default="claude-opus-5", help="Anthropic API model for the api backend")
    parser.add_argument("--backend", choices=("api", "cli"), help="api (Anthropic SDK, needs ANTHROPIC_API_KEY) or cli (local Claude Code CLI, tools disabled); default api if a key is set, else cli")
    parser.add_argument("--cli-model", default="opus")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--env-file", type=Path, help="untracked .env providing ANTHROPIC_API_KEY for code-llm")
    args = parser.parse_args()
    task = load_yaml("task.yaml")
    analysis = load_yaml("analysis.yaml")
    if args.command == "effects" and args.dry_run:
        print_plan(
            STAGE,
            [repo_path(task["runs"][r]["raw_scores"]) for r in ("discovery", "confirmatory")],
            [args.output_root / "identities.csv", args.output_root / "summary.json"],
            {"api_calls": 0},
        )
        return 0
    return {"effects": cmd_effects, "template": cmd_template, "code-llm": cmd_code_llm, "merge": cmd_merge, "contrast": cmd_contrast}[args.command](args, task, analysis)


if __name__ == "__main__":
    raise SystemExit(main())
