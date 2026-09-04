"""Frozen contrastive surface for the v9 paper-faithful probe."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalRow:
    sample_id: str
    pair_id: str
    family_id: str
    split: str
    label: int | None
    context_label: str
    variant_family: str
    cue_span_text: str
    cue_spans_json: str
    system_prompt: str
    user_prompt: str

    @property
    def messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt})
        return messages

    @property
    def cue_spans(self) -> tuple[str, ...]:
        if self.cue_spans_json.strip():
            payload = json.loads(self.cue_spans_json)
            values = payload.get(self.variant_family, [])
            if not isinstance(values, list):
                raise ValueError(f"cue spans for {self.sample_id} are not a list")
            return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))
        text = self.cue_span_text.strip()
        if not text:
            return ()
        if "||" not in text or "=" not in text:
            return (text,)
        variants = {}
        for chunk in text.split("||"):
            key, separator, value = chunk.strip().partition("=")
            if separator:
                variants[key.strip()] = value.strip()
        value = variants.get(self.variant_family, "")
        return (value,) if value else ()


def load_eval_rows(dataset_root: Path) -> list[EvalRow]:
    split_by_sample: dict[str, str] = {}
    with (dataset_root / "splits.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            if sample_id in split_by_sample:
                raise ValueError(f"duplicate split row: {sample_id}")
            split_by_sample[sample_id] = str(row["split_id"])

    output: list[EvalRow] = []
    with (dataset_root / "samples.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            context = str(raw["context_label"])
            label = {"deploy_like": 0, "eval_like": 1, "neutral_decoy": None}.get(context)
            if context not in {"deploy_like", "eval_like", "neutral_decoy"}:
                raise ValueError(f"unknown context label: {context}")
            sample_id = str(raw["sample_id"])
            output.append(
                EvalRow(
                    sample_id=sample_id,
                    pair_id=str(raw["pair_id"]),
                    family_id=str(raw["family_id"]),
                    split=split_by_sample[sample_id],
                    label=label,
                    context_label=context,
                    variant_family=str(raw["variant_family"]),
                    cue_span_text=str(raw.get("cue_span_text") or ""),
                    cue_spans_json=str(raw.get("cue_spans_json") or ""),
                    system_prompt=str(raw.get("system_prompt") or ""),
                    user_prompt=str(raw["user_prompt"]),
                )
            )
    if len(output) != 448 or len({row.sample_id for row in output}) != 448:
        raise ValueError("contrastive surface must contain 448 unique samples")
    expected = {
        "train": 192,
        "val": 32,
        "test": 32,
        "steering_anchor": 32,
        "holdout_hardneg": 64,
        "holdout_realism": 32,
        "holdout_mats_aux": 64,
    }
    observed = {split: sum(row.split == split for row in output) for split in expected}
    if observed != expected:
        raise ValueError(f"split counts differ from paper contract: {observed}")
    return output

