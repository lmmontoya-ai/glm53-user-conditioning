import json
from pathlib import Path

import pytest

from src.glm53_user_eval.schemas import SourceLocks
from src.glm53_user_eval.source_locks import (
    canonical_sha256,
    load_source_locks,
    validate_model_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "reference/source_locks_glm53_user_eval_v1.json"


def test_committed_source_lock_validates() -> None:
    locks = load_source_locks(LOCK_PATH)
    assert locks.model.revision == "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    assert locks.model.safetensor_shards == 62


def test_source_lock_rejects_short_commit() -> None:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    payload["transluce"]["commit"] = "abc"
    with pytest.raises(ValueError):
        SourceLocks.model_validate(payload)


def test_model_metadata_must_match_lock() -> None:
    locks = load_source_locks(LOCK_PATH)
    validate_model_metadata(
        locks,
        observed_revision=locks.model.revision,
        safetensor_shards=62,
        safetensor_bytes=328337455672,
    )
    with pytest.raises(ValueError, match="shard count"):
        validate_model_metadata(
            locks,
            observed_revision=locks.model.revision,
            safetensor_shards=61,
            safetensor_bytes=328337455672,
        )


def test_canonical_hash_ignores_dictionary_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
