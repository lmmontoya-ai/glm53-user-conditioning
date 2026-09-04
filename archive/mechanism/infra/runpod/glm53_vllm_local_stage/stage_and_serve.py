#!/usr/bin/env python3
"""Copy an immutable GLM-5.3 snapshot to Pod-local disk, then start vLLM."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

CHUNK_BYTES = 16 * 1024 * 1024
MIN_FREE_MARGIN_BYTES = 8 * 1024**3
SAFE_LOCAL_ROOT = Path("/runpod-local")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != "glm53_model_stage_v1":
        raise ValueError("unexpected model-stage manifest schema")
    weights = manifest.get("safetensor_sha256")
    if not isinstance(weights, dict) or len(weights) != 62:
        raise ValueError("model-stage manifest must contain 62 safetensor hashes")
    return manifest, hashlib.sha256(raw).hexdigest()


def _assert_safe_target(target: Path) -> None:
    resolved_root = SAFE_LOCAL_ROOT.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"MODEL_LOCAL must be a child of {resolved_root}")


def _copy_and_hash(source: Path, destination: Path, expected: str) -> None:
    temporary = destination.with_name(f".{destination.name}.partial")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        for chunk in iter(lambda: reader.read(CHUNK_BYTES), b""):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    observed = digest.hexdigest()
    if observed != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"hash mismatch for {source.name}: expected {expected}, observed {observed}"
        )
    os.replace(temporary, destination)


def _copy_non_weights(source: Path, staging: Path, weight_names: set[str]) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.name in weight_names or relative.parts[0] == ".cache":
            continue
        destination = staging / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _sentinel_matches(target: Path, manifest_hash: str, revision: str) -> bool:
    sentinel = target / ".glm53_local_stage.json"
    if not sentinel.is_file():
        return False
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    return (
        payload.get("manifest_sha256") == manifest_hash
        and payload.get("revision") == revision
        and payload.get("weight_shards") == 62
    )


def _check_free_space(target: Path, required_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < required_bytes + MIN_FREE_MARGIN_BYTES:
        raise OSError(
            f"insufficient Pod-local disk: need {required_bytes + MIN_FREE_MARGIN_BYTES}, "
            f"have {free_bytes}"
        )


def _verify_local_weights(
    root: Path, weights: dict[str, str], workers: int
) -> None:
    missing = sorted(name for name in weights if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"missing local weight shards: {missing}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        observed = list(executor.map(lambda name: _sha256(root / name), sorted(weights)))
    for name, digest in zip(sorted(weights), observed, strict=True):
        if digest != weights[name]:
            raise ValueError(
                f"hash mismatch for {name}: expected {weights[name]}, observed {digest}"
            )


def stage_model(source: Path, target: Path, manifest_path: Path, workers: int) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"MODEL_SOURCE is not a directory: {source}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MODEL_STAGE_MANIFEST is not a file: {manifest_path}")
    _assert_safe_target(target)

    manifest, manifest_hash = _load_manifest(manifest_path)
    revision = str(manifest["revision"])
    if source.name != revision:
        raise ValueError(f"source directory {source.name!r} differs from revision {revision!r}")
    if _sentinel_matches(target, manifest_hash, revision):
        print(json.dumps({"event": "local_stage_reused", "target": str(target)}), flush=True)
        return

    weights: dict[str, str] = manifest["safetensor_sha256"]
    missing = sorted(name for name in weights if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"missing source weight shards: {missing}")

    required_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    _check_free_space(target, required_bytes)

    staging = target.with_name(f".{target.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(
        json.dumps(
            {
                "event": "local_stage_started",
                "source": str(source),
                "target": str(target),
                "weight_bytes": manifest["safetensor_bytes"],
                "weight_shards": manifest["safetensor_shards"],
                "workers": workers,
            }
        ),
        flush=True,
    )
    _copy_non_weights(source, staging, set(weights))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_copy_and_hash, source / name, staging / name, expected)
            for name, expected in sorted(weights.items())
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            print(
                json.dumps(
                    {"event": "weight_shard_staged", "completed": index, "total": len(futures)}
                ),
                flush=True,
            )

    sentinel = {
        "schema_version": "glm53_local_stage_v1",
        "manifest_sha256": manifest_hash,
        "revision": revision,
        "weight_shards": len(weights),
        "weight_bytes": manifest["safetensor_bytes"],
    }
    (staging / ".glm53_local_stage.json").write_text(
        json.dumps(sentinel, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    print(json.dumps({"event": "local_stage_completed", "target": str(target)}), flush=True)


def stage_model_from_hub(
    model_id: str,
    revision: str,
    target: Path,
    manifest_path: Path,
    workers: int,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MODEL_STAGE_MANIFEST is not a file: {manifest_path}")
    _assert_safe_target(target)
    manifest, manifest_hash = _load_manifest(manifest_path)
    if manifest["revision"] != revision:
        raise ValueError("MODEL_REVISION differs from the verified stage manifest")
    if _sentinel_matches(target, manifest_hash, revision):
        print(json.dumps({"event": "local_stage_reused", "target": str(target)}), flush=True)
        return

    _check_free_space(target, int(manifest["safetensor_bytes"]))
    staging = target.with_name(f".{target.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(
        json.dumps(
            {
                "event": "hub_stage_started",
                "model_id": model_id,
                "revision": revision,
                "target": str(target),
                "weight_bytes": manifest["safetensor_bytes"],
                "weight_shards": manifest["safetensor_shards"],
                "download_workers": workers,
            }
        ),
        flush=True,
    )
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=staging,
        max_workers=workers,
    )
    print(json.dumps({"event": "hub_download_completed"}), flush=True)
    weights: dict[str, str] = manifest["safetensor_sha256"]
    _verify_local_weights(staging, weights, workers)
    print(json.dumps({"event": "local_hash_verification_completed"}), flush=True)
    shutil.rmtree(staging / ".cache", ignore_errors=True)
    sentinel = {
        "schema_version": "glm53_local_stage_v1",
        "manifest_sha256": manifest_hash,
        "revision": revision,
        "weight_shards": len(weights),
        "weight_bytes": manifest["safetensor_bytes"],
        "download_source": model_id,
    }
    (staging / ".glm53_local_stage.json").write_text(
        json.dumps(sentinel, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    print(json.dumps({"event": "local_stage_completed", "target": str(target)}), flush=True)


def main() -> None:
    manifest_path = Path(os.environ["MODEL_STAGE_MANIFEST"])
    target = Path(
        os.environ.get(
            "MODEL_LOCAL",
            "/runpod-local/GLM-5.3-Flash/04c4e9e95c5da8862dced7e5056455116f83a7e0",
        )
    )
    workers = int(os.environ.get("STAGE_WORKERS", "4"))
    if workers < 1 or workers > 8:
        raise ValueError("STAGE_WORKERS must be between 1 and 8")
    model_id = os.environ.get("MODEL_ID")
    revision = os.environ.get("MODEL_REVISION")
    source_value = os.environ.get("MODEL_SOURCE")
    if model_id and revision:
        stage_model_from_hub(model_id, revision, target, manifest_path, workers)
    elif source_value:
        stage_model(Path(source_value), target, manifest_path, workers)
    else:
        raise ValueError("set MODEL_ID and MODEL_REVISION, or set MODEL_SOURCE")
    command = ["vllm", "serve", str(target), *sys.argv[1:]]
    print(json.dumps({"event": "vllm_exec", "argv": command}), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
