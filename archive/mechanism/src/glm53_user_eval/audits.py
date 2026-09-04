"""Artifact, budget, and secret checks."""

from __future__ import annotations

from pathlib import Path


SECRET_MARKERS = (
    "rpa_",
    "aws_secret_access_key",
    "aws_access_key_id",
    "-----begin openssh private key-----",
    "-----begin rsa private key-----",
    "zai_api_key=",
    "openrouter_api_key=",
    "hf_token=",
    "auth.json",
)


def reject_secret_text(text: str) -> None:
    lowered = text.casefold()
    found = [marker for marker in SECRET_MARKERS if marker in lowered]
    if found:
        raise ValueError(f"credential-like material found: {found}")


def audit_tree_for_secrets(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            reject_secret_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        except ValueError:
            findings.append(str(path.relative_to(root)))
    return findings


def projected_budget_ok(spent_usd: float, incremental_usd: float, hard_cap_usd: float) -> bool:
    if min(spent_usd, incremental_usd, hard_cap_usd) < 0:
        raise ValueError("budget values cannot be negative")
    return spent_usd + incremental_usd <= hard_cap_usd


def covered_by_manifest(root: Path, manifest_paths: set[str]) -> bool:
    evidence = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    return evidence == manifest_paths
