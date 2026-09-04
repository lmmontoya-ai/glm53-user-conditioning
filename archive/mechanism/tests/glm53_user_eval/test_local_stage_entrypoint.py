from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "infra"
    / "runpod"
    / "glm53_vllm_local_stage"
    / "stage_and_serve.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("glm53_stage_and_serve", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_requires_all_weight_shards(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "model_stage.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "glm53_model_stage_v1",
                "safetensor_sha256": {"only-one.safetensors": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    try:
        module._load_manifest(manifest)
    except ValueError as error:
        assert "62 safetensor hashes" in str(error)
    else:
        raise AssertionError("incomplete weight manifest was accepted")


def test_copy_and_hash_is_atomic(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"immutable model bytes")
    expected = module._sha256(source)
    module._copy_and_hash(source, destination, expected)
    assert destination.read_bytes() == source.read_bytes()
    assert not (tmp_path / ".destination.bin.partial").exists()


def test_local_weight_verification_rejects_mismatch(tmp_path: Path) -> None:
    module = _module()
    for index in range(62):
        (tmp_path / f"model-{index:05d}.safetensors").write_bytes(str(index).encode())
    weights = {
        f"model-{index:05d}.safetensors": module._sha256(
            tmp_path / f"model-{index:05d}.safetensors"
        )
        for index in range(62)
    }
    weights["model-00000.safetensors"] = "0" * 64
    try:
        module._verify_local_weights(tmp_path, weights, workers=2)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("a local weight hash mismatch was accepted")
