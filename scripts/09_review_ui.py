"""Local review server for the human-review steps.

Serves src/glm53/review_ui/index.html and a few JSON endpoints. This script is the only thing
that touches disk. Standard library only.

  uv run python scripts/09_review_ui.py            # prints the URL
  uv run python scripts/09_review_ui.py --port 0   # free port
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "src/glm53/review_ui/index.html"

MODES = ("mode1", "mode2", "mode3", "mode4")
MODE1_VERDICTS = {"1": "correct", "2": "answer_wrong", "3": "confidence_wrong", "4": "both_wrong"}
MODE4_VERDICTS = {"1": "agree", "2": "disagree_evaluation", "3": "disagree_deference", "4": "disagree_both"}
CATEGORIES = ("scrutiny", "capabilities", "business", "other")
TEMPLATE_COLUMNS = ["persona_key", "name", "role_text", "category", "ambiguous", "alternative_category", "notes"]
MODE3_TAGS = ["hedges", "defers to user", "mentions user", "mentions being evaluated", "refuses or deflects", "longer justification", "other"]
GROUP_LABELS = {"famous_ai": "famous AI", "famous_ai_real": "famous AI (published address)", "unknown_ai": "unknown AI twin", "famous_nonai": "famous non-AI", "genpop": "general population"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    tmp.replace(path)


def load_taxonomy(path: Path) -> dict[str, Any]:
    """Category definitions and decision rules; uses PyYAML when present, else a minimal reader."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return {"categories": {k: " ".join(str(v["definition"]).split()) for k, v in data["categories"].items()}, "decision_rules": [str(r) for r in data.get("decision_rules", [])]}
    except ImportError:
        pass
    categories: dict[str, str] = {}
    rules: list[str] = []
    current = None
    section = None
    for line in text.splitlines():
        if line.startswith("categories:"):
            section = "categories"
            continue
        if line.startswith("decision_rules:"):
            section = "rules"
            continue
        if section == "categories":
            m = re.match(r"^  (\w+):\s*$", line)
            if m:
                current = m.group(1)
                categories[current] = ""
                continue
            if current and line.startswith("      "):
                categories[current] = (categories[current] + " " + line.strip()).strip()
        elif section == "rules":
            if line.startswith("  - "):
                rules.append(line[4:].strip())
            elif not line.startswith(" "):
                section = None
    return {"categories": categories, "decision_rules": rules}


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    if not a or len(a) != len(b):
        return None
    labels = sorted({*a, *b}, key=str)
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


class Store:
    """All disk access for the review tool lives here."""

    def __init__(self, root: Path, seed: int = 20260903) -> None:
        self.root = root
        self.seed = seed
        self.lock = threading.Lock()
        self.review_dir = root / "outputs/review"
        self.state_path = self.review_dir / "state.json"
        self.state = read_json(self.state_path, None) or {"timers": {m: 0.0 for m in MODES}, "done": {m: False for m in MODES}, "skipped": {"mode4": False}, "position": {m: 0 for m in MODES}, "last_heartbeat": {}}
        self.mode1_packet = root / "data/audits/confirmatory_40/manual_packet.jsonl"
        self.mode1_out = root / "outputs/audits/confirmatory_40_human.csv"
        self.mode1_summary = root / "outputs/audits/confirmatory_40_summary.json"
        self.mode4_packet = root / "data/audits/decomposition_160/human_audit_packet.jsonl"
        self.mode4_out = root / "outputs/audits/decomposition_160_human.csv"
        self.mode4_summary = root / "outputs/audits/decomposition_160_summary.json"
        self.roster_path = root / "data/transcripts/personas2.json"
        self.taxonomy_path = root / "configs/roles/taxonomy.yaml"
        self.template_path = root / "outputs/roles/human_coding_template.csv"
        self.human_path = root / "outputs/roles/human_coding.csv"
        self.llm_path = root / "outputs/roles/llm_coding.csv"
        self.merged_path = root / "outputs/roles/merged_coding.csv"
        self.sets_full = root / "outputs/transcripts/sample/sets_full.json"
        self.transcripts_path = root / "data/transcripts/v7_transcripts.jsonl"
        self.identities_path = root / "outputs/identities/identities.csv"
        self.mode3_private = self.review_dir / "mode3_quartets_private.json"
        self.mode3_out = root / "outputs/transcripts/reading_notes.json"
        self._mode3_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------- state
    def save_state(self) -> None:
        write_json(self.state_path, self.state)

    def heartbeat(self, mode: str) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            last = self.state["last_heartbeat"].get(mode)
            if last is not None:
                delta = now - float(last)
                if 0 < delta <= 20:
                    self.state["timers"][mode] = float(self.state["timers"].get(mode, 0.0)) + delta
            self.state["last_heartbeat"][mode] = now
            self.save_state()
            return {"timers": self.state["timers"]}

    def set_position(self, mode: str, index: int) -> None:
        with self.lock:
            self.state["position"][mode] = int(index)
            self.save_state()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "timers": self.state["timers"],
                "done": self.state["done"],
                "skipped": self.state["skipped"],
                "position": self.state["position"],
                "counts": {"mode1": self._count(self.mode1_out), "mode2": self._count_human(), "mode3": self._mode3_count(), "mode4": self._count(self.mode4_out)},
                "totals": {"mode1": self._total(self.mode1_packet), "mode2": len(self._famous_roster()) if self.roster_path.exists() else 0, "mode3": None, "mode4": self._total(self.mode4_packet)},
            }

    def _total(self, path: Path) -> int:
        return len(read_jsonl(path)) if path.exists() else 0

    def _count(self, path: Path) -> int:
        return len(read_csv(path))

    def _count_human(self) -> int:
        return sum(1 for r in read_csv(self.human_path) if (r.get("category") or "").strip())

    def _mode3_count(self) -> int:
        notes = read_json(self.mode3_out, {}) or {}
        return sum(1 for q in notes.get("quartets", {}).values() if q.get("note") or any(v.get("tags") for v in q.get("labels", {}).values()))

    # ------------------------------------------------------------- mode 1
    def mode1_items(self) -> dict[str, Any]:
        rows = read_jsonl(self.mode1_packet)
        done = self.state["done"]["mode1"]
        decisions = {r["row_id"]: r for r in read_csv(self.mode1_out)}
        items = []
        for row in rows:
            script = row.get("scenario_script") or []
            dilemma = next((t[1] for t in script if t[0] == "user" and t[2] == "main"), "")
            followup = next((t[1] for t in script if t[0] == "user" and t[2] == "followup"), "")
            item = {
                "row_id": row["sample_id"],
                "dilemma": dilemma,
                "followup_question": followup,
                "first_turn_raw": row.get("subject_main") or "",
                "confidence_turn_raw": row.get("subject_followup") or "",
                "extracted": {
                    "binary_answer": row.get("binary_answer"),
                    "confidence_p": row.get("confidence_p"),
                    "refused_estimate": row.get("refused_estimate"),
                    "score_value": row.get("score_value"),
                    "judge_text": row.get("judge_text"),
                    "sample_error": row.get("sample_error"),
                },
                "decision": decisions.get(row["sample_id"]),
            }
            if done:
                item["group"] = row.get("group")
                item["persona"] = row.get("persona")
            items.append(item)
        return {"items": items, "verdicts": MODE1_VERDICTS, "done": done, "position": self.state["position"]["mode1"], "source": str(self.mode1_packet.relative_to(self.root)).replace("\\", "/")}

    def mode1_decide(self, row_id: str, verdict: str, note: str) -> dict[str, Any]:
        if verdict not in MODE1_VERDICTS.values():
            raise ValueError("unknown verdict")
        with self.lock:
            rows = {r["row_id"]: r for r in read_csv(self.mode1_out)}
            rows[row_id] = {"row_id": row_id, "verdict": verdict, "note": note or "", "timestamp": now_iso()}
            order = [r["sample_id"] for r in read_jsonl(self.mode1_packet)]
            ordered = [rows[k] for k in order if k in rows] + [v for k, v in rows.items() if k not in order]
            write_csv(self.mode1_out, ["row_id", "verdict", "note", "timestamp"], ordered)
            summary = self._mode1_summary(ordered)
            write_json(self.mode1_summary, summary)
            if summary["reviewed"] >= summary["total"] and summary["total"] > 0:
                self.state["done"]["mode1"] = True
                self.save_state()
            return {"summary": summary, "done": self.state["done"]["mode1"]}

    def _mode1_summary(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        packet = {r["sample_id"]: r for r in read_jsonl(self.mode1_packet)}
        counts = {v: 0 for v in MODE1_VERDICTS.values()}
        by_group: dict[str, dict[str, int]] = {}
        disagreements = []
        for d in decisions:
            counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
            group = packet.get(d["row_id"], {}).get("group", "?")
            by_group.setdefault(group, {"reviewed": 0, "correct": 0})
            by_group[group]["reviewed"] += 1
            if d["verdict"] == "correct":
                by_group[group]["correct"] += 1
            else:
                disagreements.append({"row_id": d["row_id"], "verdict": d["verdict"], "note": d.get("note", ""), "group": group})
        total = len(packet)
        reviewed = len(decisions)
        exceptions = "; ".join(f"{x['row_id']} ({x['verdict'].replace('_', ' ')}{': ' + x['note'] if x['note'] else ''})" for x in disagreements) or "none"
        sentence = f"I read {reviewed} randomly selected, score-blind rows (8 per group); the extraction judge's answer and confidence were correct in {counts['correct']}/{total}, with the following exceptions: {exceptions}."
        return {"total": total, "reviewed": reviewed, "counts": counts, "by_group": by_group, "disagreements": disagreements, "paste_sentence": sentence, "complete": reviewed >= total, "updated_at": now_iso()}

    # ------------------------------------------------------------- mode 2
    def _assert_blind(self, path: Path) -> Path:
        """Refuse to read any file that could carry effects while coding roles."""
        rel = str(path.resolve()).replace("\\", "/").lower()
        forbidden = ("outputs/identities/", "outputs/estimands/", "outputs/reproduce/", "outputs/decompose/", "outputs/transcripts/", "identities.csv", "raw_scores", "role_contrast")
        if any(f in rel for f in forbidden):
            raise PermissionError(f"mode 2 must not read {path}")
        return path

    def _famous_roster(self) -> list[dict[str, Any]]:
        roster = read_json(self._assert_blind(self.roster_path))
        by_slug: dict[str, dict[str, Any]] = {}
        for row in roster["famous_ai"]:
            slug = str(row["key"]).removeprefix("fai2_")
            by_slug[slug] = {"persona_key": row["key"], "name": row["name"], "role_text": f"{row['affiliation']} ({row['org']})", "published_role_text": None, "published_key": None}
        for row in roster.get("famous_ai_real", []):
            slug = str(row["key"]).removeprefix("fai2r_")
            if slug in by_slug:
                text = f"{row['affiliation']} ({row['org']})"
                by_slug[slug]["published_key"] = row["key"]
                if text != by_slug[slug]["role_text"]:
                    by_slug[slug]["published_role_text"] = text
        return list(by_slug.values())

    def mode2_items(self) -> dict[str, Any]:
        roster = self._famous_roster()
        taxonomy = load_taxonomy(self._assert_blind(self.taxonomy_path))
        template = read_csv(self._assert_blind(self.template_path)) if self.template_path.exists() else []
        template_order = [r["persona_key"] for r in template] or [r["persona_key"] for r in roster]
        human = {r["persona_key"]: r for r in read_csv(self._assert_blind(self.human_path))} if self.human_path.exists() else {}
        rng = random.Random(f"{self.seed}|mode2")
        order = list(template_order)
        rng.shuffle(order)
        by_key = {r["persona_key"]: r for r in roster}
        items = []
        for key in order:
            r = by_key.get(key)
            if r is None:
                continue
            h = human.get(key, {})
            items.append({**r, "decision": {"category": h.get("category", ""), "ambiguous": str(h.get("ambiguous", "")).lower() == "true", "alternative_category": h.get("alternative_category", ""), "notes": h.get("notes", "")} if h else None})
        loaded = [str(p.relative_to(self.root)).replace("\\", "/") for p in (self.roster_path, self.taxonomy_path, self.template_path, self.human_path) if p.exists()]
        return {"items": items, "taxonomy": taxonomy, "categories": list(CATEGORIES), "loaded_files": loaded, "done": self.state["done"]["mode2"], "position": self.state["position"]["mode2"], "template_columns": TEMPLATE_COLUMNS}

    def mode2_decide(self, persona_key: str, category: str, ambiguous: bool, alternative: str, notes: str) -> dict[str, Any]:
        if category and category not in CATEGORIES:
            raise ValueError("unknown category")
        if alternative and alternative not in CATEGORIES:
            raise ValueError("unknown alternative category")
        with self.lock:
            template = read_csv(self._assert_blind(self.template_path)) if self.template_path.exists() else []
            if not template:
                template = [{"persona_key": r["persona_key"], "name": r["name"], "role_text": r["role_text"], "category": "", "ambiguous": "", "alternative_category": "", "notes": ""} for r in self._famous_roster()]
            existing = {r["persona_key"]: r for r in read_csv(self.human_path)} if self.human_path.exists() else {}
            rows = []
            for t in template:
                row = dict(existing.get(t["persona_key"], t))
                row["name"], row["role_text"] = t["name"], t["role_text"]
                if t["persona_key"] == persona_key:
                    row["category"] = category
                    row["ambiguous"] = "True" if ambiguous else "False"
                    row["alternative_category"] = alternative or ""
                    row["notes"] = notes or ""
                rows.append(row)
            write_csv(self.human_path, TEMPLATE_COLUMNS, rows)
            complete = all((r.get("category") or "").strip() for r in rows)
            self.state["done"]["mode2"] = complete
            self.save_state()
            return {"coded": sum(1 for r in rows if (r.get("category") or "").strip()), "total": len(rows), "done": complete}

    def mode2_merge(self) -> dict[str, Any]:
        script = self.root / "scripts/04_identities.py"
        if not script.exists():
            return {"returncode": -1, "stdout": "", "stderr": f"missing {script}"}
        completed = subprocess.run([sys.executable, "-X", "utf8", str(script), "merge"], cwd=str(self.root), capture_output=True, text=True, encoding="utf-8")
        return {"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}

    def mode2_disagreements(self) -> dict[str, Any]:
        merged = read_csv(self._assert_blind(self.merged_path)) if self.merged_path.exists() else []
        llm = {r["persona_key"]: r for r in read_csv(self.llm_path)} if self.llm_path.exists() else {}
        human = {r["persona_key"]: r for r in read_csv(self.human_path)} if self.human_path.exists() else {}
        items = []
        for r in merged:
            if str(r.get("agree", "")).lower() == "true":
                continue
            items.append({
                "persona_key": r["persona_key"],
                "name": r.get("name", ""),
                "role_text": r.get("role_text", ""),
                "llm_category": r.get("llm_category", ""),
                "llm_ambiguous": r.get("llm_ambiguous", ""),
                "llm_alternative": r.get("llm_alternative", ""),
                "llm_justification": llm.get(r["persona_key"], {}).get("justification", ""),
                "human_category": r.get("human_category", ""),
                "human_ambiguous": r.get("human_ambiguous", ""),
                "human_alternative": r.get("human_alternative", ""),
                "human_notes": human.get(r["persona_key"], {}).get("notes", ""),
                "final_category": r.get("final_category", ""),
                "resolution": r.get("resolution", ""),
            })
        return {"items": items, "merged_exists": self.merged_path.exists(), "unresolved": sum(1 for i in items if not i["final_category"] or not i["resolution"])}

    def mode2_resolve(self, persona_key: str, final_category: str, resolution: str) -> dict[str, Any]:
        if final_category not in CATEGORIES:
            raise ValueError("unknown category")
        with self.lock:
            with self.merged_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
            found = False
            for r in rows:
                if r["persona_key"] == persona_key:
                    r["final_category"] = final_category
                    r["resolution"] = resolution or f"adjudicated in review UI {now_iso()}"
                    found = True
            if not found:
                raise KeyError(persona_key)
            write_csv(self.merged_path, columns, rows)
        return self.mode2_disagreements()

    # ------------------------------------------------------------- mode 3
    def _role_source(self) -> tuple[dict[str, str], str]:
        """Role labels for choosing scrutiny identities: human coding, overridden by adjudicated finals; LLM sheet only as a fallback."""
        labels: dict[str, str] = {}
        source = "none"
        if self.human_path.exists():
            labels = {r["persona_key"]: r["category"] for r in read_csv(self.human_path) if r.get("category")}
            if labels:
                source = "outputs/roles/human_coding.csv"
        if labels and self.merged_path.exists():
            finals = {r["persona_key"]: r["final_category"] for r in read_csv(self.merged_path) if r.get("final_category")}
            if finals:
                labels.update(finals)
                source = "outputs/roles/human_coding.csv + outputs/roles/merged_coding.csv finals"
        if not labels and self.llm_path.exists():
            labels = {r["persona_key"]: r["category"] for r in read_csv(self.llm_path)}
            source = "outputs/roles/llm_coding.csv (human coding not available yet)"
        return labels, source

    def _build_mode3(self) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}|mode3")
        quartets: list[dict[str, Any]] = []
        notes: list[str] = []
        existing = read_json(self.sets_full, []) or []
        for item in existing:
            quartets.append({"quartet_id": item["set_id"], "source": "stage5_sample", "stimulus_id": item["stimulus_id"], "members": item["members"]})
        labels, role_source = self._role_source()
        extra_needed = 10
        if not self.identities_path.exists() or not self.transcripts_path.exists() or not labels:
            notes.append(f"additional quartets not built: identities={self.identities_path.exists()}, transcripts={self.transcripts_path.exists()}, role labels from {role_source}")
        else:
            identities = read_csv(self.identities_path)
            scrutiny = [r for r in identities if r.get("profile") == "constructed" and labels.get(r["persona_key"]) == "scrutiny" and r.get("confirmatory_twin_adjusted_pp")]
            scrutiny.sort(key=lambda r: float(r["confirmatory_twin_adjusted_pp"]))
            chosen = scrutiny[:extra_needed]
            if not chosen:
                notes.append(f"no scrutiny-role identities found in {role_source}")
            wanted_keys = {r["persona_key"] for r in chosen} | {r["twin_key"] for r in chosen}
            used_dilemmas = {q["stimulus_id"] for q in quartets}
            rows_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
            group_rows: dict[str, dict[str, list[dict[str, Any]]]] = {"famous_nonai": {}, "genpop": {}}
            with self.transcripts_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if not row.get("parse_valid"):
                        continue
                    if row["persona_key"] in wanted_keys:
                        rows_by_cell[(row["persona_key"], row["stimulus_id"])] = row
                    elif row["group"] in group_rows:
                        group_rows[row["group"]].setdefault(row["stimulus_id"], []).append(row)
            for ident in chosen:
                candidates = [s for s in sorted({k[1] for k in rows_by_cell}) if s not in used_dilemmas and (ident["persona_key"], s) in rows_by_cell and (ident["twin_key"], s) in rows_by_cell and group_rows["famous_nonai"].get(s) and group_rows["genpop"].get(s)]
                if not candidates:
                    notes.append(f"no complete dilemma for {ident['persona_key']}")
                    continue
                stimulus = rng.choice(candidates)
                used_dilemmas.add(stimulus)
                members = [rows_by_cell[(ident["persona_key"], stimulus)], rows_by_cell[(ident["twin_key"], stimulus)], rng.choice(group_rows["famous_nonai"][stimulus]), rng.choice(group_rows["genpop"][stimulus])]
                quartets.append({"quartet_id": f"extra_{len(quartets) - len(existing) + 1:02d}", "source": f"scrutiny_top10 ({role_source})", "stimulus_id": stimulus, "members": members})
        blinded = []
        private = []
        for q in quartets:
            labels_abcd = ["A", "B", "C", "D"]
            order = list(range(len(q["members"])))
            rng.shuffle(order)
            mapping = {}
            boxes = []
            for label, idx in zip(labels_abcd, order):
                m = q["members"][idx]
                mapping[label] = {"group": m["group"], "persona_key": m["persona_key"], "sample_id": m.get("sample_id")}
                boxes.append({
                    "label": label,
                    "first_turn_answer": self._redact(m.get("first_turn_answer", ""), m),
                    "confidence_turn_answer": self._redact(m.get("confidence_turn_answer", ""), m),
                    "first_turn_choice": m.get("first_turn_choice"),
                    "folded_confidence": m.get("folded_confidence"),
                    "first_turn_visible_tokens": (m.get("first_turn_usage") or {}).get("visible_tokens"),
                    "confidence_turn_visible_tokens": (m.get("confidence_turn_usage") or {}).get("visible_tokens"),
                })
            blinded.append({"quartet_id": q["quartet_id"], "stimulus_id": q["stimulus_id"], "dilemma": q["members"][0].get("dilemma_text", ""), "boxes": boxes})
            private.append({"quartet_id": q["quartet_id"], "source": q["source"], "stimulus_id": q["stimulus_id"], "mapping": mapping})
        payload = {"built_at": now_iso(), "seed": self.seed, "role_source": role_source, "notes": notes, "blinded": blinded, "private": private}
        write_json(self.mode3_private, payload)
        return payload

    @staticmethod
    def _redact(text: str, row: dict[str, Any]) -> str:
        out = text or ""
        for key, placeholder in (("email", "<USER_AFFILIATION>"), ("affiliation", "<USER_AFFILIATION>"), ("organization", "<USER_AFFILIATION>"), ("name", "<USER_NAME>")):
            literal = str(row.get(key) or "").strip()
            if literal:
                out = re.sub(re.escape(literal), placeholder, out, flags=re.IGNORECASE)
        return out

    def mode3_items(self) -> dict[str, Any]:
        if self._mode3_cache is None:
            cached = read_json(self.mode3_private, None)
            notes_file = read_json(self.mode3_out, {}) or {}
            # Rebuild the quartets when the role labels have improved (LLM fallback -> human coding)
            # and nothing has been read yet; once notes exist the quartet ids must stay fixed.
            stale = bool(cached) and not notes_file.get("quartets") and cached.get("role_source") != self._role_source()[1]
            self._mode3_cache = cached if cached and not stale else self._build_mode3()
        data = self._mode3_cache
        notes_file = read_json(self.mode3_out, {}) or {}
        revealed = bool(notes_file.get("revealed"))
        items = []
        for q in data["blinded"]:
            saved = (notes_file.get("quartets") or {}).get(q["quartet_id"], {})
            items.append({**q, "saved": saved})
        return {"items": items, "tags": MODE3_TAGS, "notes_built": data["notes"], "revealed": revealed, "done": self.state["done"]["mode3"], "position": self.state["position"]["mode3"], "reveal": notes_file.get("reveal") if revealed else None}

    def mode3_save(self, quartet_id: str, label: str | None, tags: list[str] | None, note: str | None) -> dict[str, Any]:
        with self.lock:
            notes_file = read_json(self.mode3_out, None) or {"schema_version": "glm53_review_reading_notes_v1", "revealed": False, "quartets": {}}
            q = notes_file["quartets"].setdefault(quartet_id, {"labels": {}, "note": ""})
            if label:
                q["labels"].setdefault(label, {})["tags"] = [t for t in (tags or []) if t in MODE3_TAGS]
            if note is not None:
                q["note"] = note
            q["updated_at"] = now_iso()
            write_json(self.mode3_out, notes_file)
            return {"saved": q}

    def mode3_reveal(self) -> dict[str, Any]:
        with self.lock:
            data = self._mode3_cache or read_json(self.mode3_private, None) or self._build_mode3()
            notes_file = read_json(self.mode3_out, None) or {"schema_version": "glm53_review_reading_notes_v1", "revealed": False, "quartets": {}}
            mapping = {p["quartet_id"]: p for p in data["private"]}
            tag_table: dict[str, dict[str, int]] = {}
            n_boxes: dict[str, int] = {}
            all_notes = []
            for qid, q in notes_file["quartets"].items():
                priv = mapping.get(qid)
                if priv is None:
                    continue
                for label, entry in q.get("labels", {}).items():
                    true = priv["mapping"].get(label, {})
                    entry["group"] = true.get("group")
                    entry["persona_key"] = true.get("persona_key")
                    g = true.get("group", "?")
                    n_boxes[g] = n_boxes.get(g, 0) + 1
                    for t in entry.get("tags", []):
                        tag_table.setdefault(t, {})
                        tag_table[t][g] = tag_table[t].get(g, 0) + 1
                q["source"] = priv["source"]
                if q.get("note"):
                    all_notes.append(f"{qid}: {q['note']}")
            n_read = sum(1 for q in notes_file["quartets"].values() if q.get("note") or any(v.get("tags") for v in q.get("labels", {}).values()))
            dilemmas = {mapping[qid]["stimulus_id"] for qid in notes_file["quartets"] if qid in mapping}
            skeleton = (f"I read {n_read} matched quartets across {len(dilemmas)} dilemmas. Differences I could see: [my notes]. What I could not see: [my notes].\n\nMy notes, verbatim:\n" + "\n".join(f"- {n}" for n in all_notes))
            notes_file["revealed"] = True
            notes_file["reveal"] = {"revealed_at": now_iso(), "tags_by_true_group": tag_table, "boxes_by_true_group": n_boxes, "quartets_read": n_read, "dilemmas": sorted(dilemmas), "skeleton": skeleton, "role_source_for_extra_quartets": data["role_source"], "build_notes": data["notes"]}
            write_json(self.mode3_out, notes_file)
            self.state["done"]["mode3"] = True
            self.save_state()
            return notes_file["reveal"]

    # ------------------------------------------------------------- mode 4
    def mode4_items(self) -> dict[str, Any]:
        rows = read_jsonl(self.mode4_packet)
        done = self.state["done"]["mode4"]
        decisions = {r["audit_id"]: r for r in read_csv(self.mode4_out)}
        items = []
        for row in rows:
            item = {
                "audit_id": row["audit_id"],
                "dilemma": row.get("dilemma", ""),
                "first_turn_reasoning": row.get("first_turn_reasoning", ""),
                "first_turn_answer": row.get("first_turn_answer", ""),
                "confidence_turn_reasoning": row.get("confidence_turn_reasoning", ""),
                "confidence_turn_answer": row.get("confidence_turn_answer", ""),
                "judges": {
                    name: {
                        "explicit_evaluation": (row.get(f"{name}_annotation") or {}).get("explicit_evaluation_or_scrutiny_inference"),
                        "deference": (row.get(f"{name}_annotation") or {}).get("deference_to_user_expertise"),
                        "rationale": (row.get(f"{name}_annotation") or {}).get("brief_rationale", ""),
                    }
                    for name in ("luna", "terra")
                },
                "decision": decisions.get(row["audit_id"]),
            }
            if done:
                item["system_profile"] = row.get("system_profile")
            items.append(item)
        return {"items": items, "verdicts": MODE4_VERDICTS, "done": done, "skipped": self.state["skipped"]["mode4"], "position": self.state["position"]["mode4"], "source": str(self.mode4_packet.relative_to(self.root)).replace("\\", "/")}

    def mode4_decide(self, audit_id: str, verdict: str, note: str) -> dict[str, Any]:
        if verdict not in MODE4_VERDICTS.values():
            raise ValueError("unknown verdict")
        with self.lock:
            rows = {r["audit_id"]: r for r in read_csv(self.mode4_out)}
            rows[audit_id] = {"audit_id": audit_id, "verdict": verdict, "note": note or "", "timestamp": now_iso()}
            order = [r["audit_id"] for r in read_jsonl(self.mode4_packet)]
            ordered = [rows[k] for k in order if k in rows]
            write_csv(self.mode4_out, ["audit_id", "verdict", "note", "timestamp"], ordered)
            summary = self._mode4_summary(ordered)
            write_json(self.mode4_summary, summary)
            if summary["reviewed"] >= summary["total"] and summary["total"] > 0:
                self.state["done"]["mode4"] = True
                self.save_state()
            return {"summary": summary, "done": self.state["done"]["mode4"]}

    def _mode4_summary(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        packet = {r["audit_id"]: r for r in read_jsonl(self.mode4_packet)}
        counts = {v: 0 for v in MODE4_VERDICTS.values()}
        kappa: dict[str, Any] = {}
        human_eval, human_def = [], []
        judge_eval = {"luna": [], "terra": []}
        judge_def = {"luna": [], "terra": []}
        disagreements = []
        for d in decisions:
            counts[d["verdict"]] += 1
            row = packet.get(d["audit_id"], {})
            luna = row.get("luna_annotation") or {}
            terra = row.get("terra_annotation") or {}
            dis_eval = d["verdict"] in ("disagree_evaluation", "disagree_both")
            dis_def = d["verdict"] in ("disagree_deference", "disagree_both")
            # The human label is defined relative to the first judge's label: agreement adopts it,
            # disagreement flips the binary label or marks the ordinal label as "other".
            le = luna.get("explicit_evaluation_or_scrutiny_inference")
            human_eval.append((not le) if dis_eval else le)
            ld = luna.get("deference_to_user_expertise")
            human_def.append("other" if dis_def else ld)
            for name, ann in (("luna", luna), ("terra", terra)):
                judge_eval[name].append(ann.get("explicit_evaluation_or_scrutiny_inference"))
                judge_def[name].append(ann.get("deference_to_user_expertise"))
            if d["verdict"] != "agree":
                disagreements.append({"audit_id": d["audit_id"], "verdict": d["verdict"], "note": d.get("note", "")})
        for name in ("luna", "terra"):
            kappa[name] = {
                "evaluation_language_kappa": cohen_kappa(human_eval, judge_eval[name]),
                "deference_kappa": cohen_kappa(human_def, judge_def[name]),
                "evaluation_agreement": (sum(a == b for a, b in zip(human_eval, judge_eval[name])) / len(human_eval)) if human_eval else None,
                "deference_agreement": (sum(a == b for a, b in zip(human_def, judge_def[name])) / len(human_def)) if human_def else None,
            }
        return {"total": len(packet), "reviewed": len(decisions), "counts": counts, "kappa_vs_judges": kappa, "kappa_definition": "human label = first judge's label when the verdict agrees; flipped (binary) or 'other' (ordinal) when it disagrees", "disagreements": disagreements, "complete": len(decisions) >= len(packet), "updated_at": now_iso()}

    def mode4_skip(self, skipped: bool) -> dict[str, Any]:
        with self.lock:
            self.state["skipped"]["mode4"] = bool(skipped)
            self.save_state()
            return {"skipped": self.state["skipped"]["mode4"]}

    # ------------------------------------------------------------- summary
    def write_summary(self) -> dict[str, Any]:
        st = self.status()
        hours = {m: st["timers"][m] / 3600.0 for m in MODES}
        m1 = read_json(self.mode1_summary, None)
        m4 = read_json(self.mode4_summary, None)
        m3 = read_json(self.mode3_out, None) or {}
        human = read_csv(self.human_path)
        merged = read_csv(self.merged_path)
        lines = ["# Human review summary", "", f"Written {now_iso()} by scripts/09_review_ui.py.", "", "## Hours", ""]
        lines.append("| Mode | Hours | Items reviewed | Complete |")
        lines.append("|---|---:|---:|---|")
        names = {"mode1": "Extraction audit (40 rows)", "mode2": "Role coding (70 identities)", "mode3": "Matched transcript reading", "mode4": "Decomposition packet (160 rows, optional)"}
        totals = {"mode1": st["totals"]["mode1"], "mode2": st["totals"]["mode2"], "mode3": len((m3.get("quartets") or {})) if m3 else 0, "mode4": st["totals"]["mode4"]}
        for m in MODES:
            complete = "yes" if st["done"][m] else ("skipped" if m == "mode4" and st["skipped"]["mode4"] else "no")
            reviewed = st["counts"][m]
            total = totals[m]
            lines.append(f"| {names[m]} | {hours[m]:.2f} | {reviewed}{' of ' + str(total) if total else ''} | {complete} |")
        lines += ["", "## Mode 1: extraction audit", ""]
        if m1:
            lines.append(f"Reviewed {m1['reviewed']} of {m1['total']} rows. Counts: " + ", ".join(f"{k} {v}" for k, v in m1["counts"].items()) + ".")
            if not m1["complete"]:
                lines.append(f"Not completed: {m1['reviewed']} of {m1['total']} rows reviewed.")
            if m1["disagreements"]:
                lines.append("Disagreements:")
                lines += [f"- {d['row_id']} ({d['group']}): {d['verdict']}" + (f" - {d['note']}" if d['note'] else "") for d in m1["disagreements"]]
            lines += ["", "Paste-ready sentence:", "", f"> {m1['paste_sentence']}"]
        else:
            lines.append("Not started.")
        lines += ["", "## Mode 2: role coding", ""]
        coded = sum(1 for r in human if (r.get("category") or "").strip())
        lines.append(f"Coded {coded} of {len(human)} identities." if human else "Not started.")
        if merged:
            dis = [r for r in merged if str(r.get("agree", "")).lower() != "true"]
            unresolved = [r for r in dis if not r.get("final_category") or not r.get("resolution")]
            lines.append(f"Merged with the LLM sheet: {len(dis)} disagreements, {len(unresolved)} unresolved.")
            for r in dis:
                lines.append(f"- {r['persona_key']}: LLM {r.get('llm_category')}, human {r.get('human_category')}, final {r.get('final_category') or '(open)'}")
        lines += ["", "## Mode 3: matched transcript reading", ""]
        if m3 and m3.get("quartets"):
            n_read = sum(1 for q in m3["quartets"].values() if q.get("note") or any(v.get("tags") for v in q.get("labels", {}).values()))
            lines.append(f"Quartets with tags or notes: {n_read}. Revealed: {m3.get('revealed', False)}." + ("" if st["done"]["mode3"] else " Not completed."))
            if m3.get("revealed"):
                rv = m3["reveal"]
                lines.append("")
                lines.append("Tags by true group (count of boxes):")
                lines.append("")
                groups = sorted({g for t in rv["tags_by_true_group"].values() for g in t})
                lines.append("| Tag | " + " | ".join(GROUP_LABELS.get(g, g) for g in groups) + " |")
                lines.append("|---|" + "---:|" * len(groups))
                for tag, counts in rv["tags_by_true_group"].items():
                    lines.append(f"| {tag} | " + " | ".join(str(counts.get(g, 0)) for g in groups) + " |")
                lines += ["", "Paragraph skeleton (my words and counts only):", "", *[f"> {l}" for l in rv["skeleton"].splitlines()]]
        else:
            lines.append("Not started.")
        lines += ["", "## Mode 4: decomposition packet (optional)", ""]
        if m4:
            lines.append(f"Reviewed {m4['reviewed']} of {m4['total']} rows. Counts: " + ", ".join(f"{k} {v}" for k, v in m4["counts"].items()) + ".")
            if not m4["complete"]:
                lines.append(f"Not completed: {m4['reviewed']} of {m4['total']} rows reviewed.")
            for name, k in m4["kappa_vs_judges"].items():
                lines.append(f"- Against {name}: evaluation-language kappa {k['evaluation_language_kappa'] if k['evaluation_language_kappa'] is None else round(k['evaluation_language_kappa'], 3)}, deference kappa {k['deference_kappa'] if k['deference_kappa'] is None else round(k['deference_kappa'], 3)}.")
            lines.append(f"Kappa definition: {m4['kappa_definition']}.")
            if m4["disagreements"]:
                lines.append("Disagreements:")
                lines += [f"- {d['audit_id']}: {d['verdict']}" + (f" - {d['note']}" if d['note'] else "") for d in m4["disagreements"]]
        elif st["skipped"]["mode4"]:
            lines.append("Skipped.")
        else:
            lines.append("Not started.")
        path = self.review_dir / "summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"path": str(path.relative_to(self.root)).replace("\\", "/"), "hours": hours}


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        return

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html")
            elif path == "/api/status":
                self._send(200, self.store.status())
            elif path == "/api/mode1/items":
                self._send(200, self.store.mode1_items())
            elif path == "/api/mode2/items":
                self._send(200, self.store.mode2_items())
            elif path == "/api/mode2/disagreements":
                self._send(200, self.store.mode2_disagreements())
            elif path == "/api/mode3/items":
                self._send(200, self.store.mode3_items())
            elif path == "/api/mode4/items":
                self._send(200, self.store.mode4_items())
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            body = self._body()
            if path == "/api/heartbeat":
                self._send(200, self.store.heartbeat(str(body["mode"])))
            elif path == "/api/position":
                self.store.set_position(str(body["mode"]), int(body["index"]))
                self._send(200, {"ok": True})
            elif path == "/api/mode1/decision":
                self._send(200, self.store.mode1_decide(str(body["row_id"]), str(body["verdict"]), str(body.get("note", ""))))
            elif path == "/api/mode2/decision":
                self._send(200, self.store.mode2_decide(str(body["persona_key"]), str(body.get("category", "")), bool(body.get("ambiguous", False)), str(body.get("alternative_category", "")), str(body.get("notes", ""))))
            elif path == "/api/mode2/merge":
                self._send(200, self.store.mode2_merge())
            elif path == "/api/mode2/resolve":
                self._send(200, self.store.mode2_resolve(str(body["persona_key"]), str(body["final_category"]), str(body.get("resolution", ""))))
            elif path == "/api/mode3/save":
                self._send(200, self.store.mode3_save(str(body["quartet_id"]), body.get("label"), body.get("tags"), body.get("note")))
            elif path == "/api/mode3/reveal":
                self._send(200, self.store.mode3_reveal())
            elif path == "/api/mode4/decision":
                self._send(200, self.store.mode4_decide(str(body["audit_id"]), str(body["verdict"]), str(body.get("note", ""))))
            elif path == "/api/mode4/skip":
                self._send(200, self.store.mode4_skip(bool(body.get("skipped", True))))
            elif path == "/api/summary":
                self._send(200, self.store.write_summary())
            else:
                self._send(404, {"error": "not found"})
        except (KeyError, ValueError, PermissionError) as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def make_server(root: Path, port: int, seed: int = 20260903) -> ThreadingHTTPServer:
    store = Store(root, seed)
    handler = type("BoundHandler", (Handler,), {"store": store})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (inputs and outputs are resolved against it)")
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    server = make_server(args.root.resolve(), args.port, args.seed)
    host, port = server.server_address[:2]
    print(f"review UI: http://{host}:{port}/  (root {args.root.resolve()}; Ctrl+C to stop)")
    fingerprint = hashlib.sha256(PAGE.read_bytes()).hexdigest()[:12]
    print(f"page {PAGE.relative_to(REPO_ROOT)} sha256 {fingerprint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
