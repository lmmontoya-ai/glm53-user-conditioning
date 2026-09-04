from __future__ import annotations

import json

import pytest
from src.glm53_user_eval.v8.artifacts import (
    atomic_json,
    atomic_jsonl,
    completed_pair,
    hash_inputs,
    sha256_file,
)


def test_atomic_json(tmp_path) -> None:
    path = tmp_path / "x.json"
    atomic_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}


def test_atomic_jsonl(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    atomic_jsonl(path, [{"a": 1}, {"a": 2}])
    assert len(path.read_text().splitlines()) == 2


def test_completed_pair(tmp_path) -> None:
    data = tmp_path / "x.bin"
    data.write_bytes(b"abc")
    manifest = tmp_path / "x.json"
    atomic_json(manifest, {"data_sha256": sha256_file(data)})
    assert completed_pair(data, manifest)


def test_incomplete_pair(tmp_path) -> None:
    assert not completed_pair(tmp_path / "missing", tmp_path / "manifest")


def test_tampered_pair(tmp_path) -> None:
    data = tmp_path / "x"
    data.write_text("a")
    manifest = tmp_path / "m"
    atomic_json(manifest, {"data_sha256": sha256_file(data)})
    data.write_text("b")
    assert not completed_pair(data, manifest)


def test_hash_inputs_records_exact_file(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"a": 1}\n', encoding="utf-8")
    result = hash_inputs({"source": source})
    assert result["source"] == {
        "path": source.as_posix(),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def test_hash_inputs_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing gate input"):
        hash_inputs({"missing": tmp_path / "missing.json"})


def test_hash_inputs_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one hashed input"):
        hash_inputs({})
