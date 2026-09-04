"""RunPod Serverless entrypoint for the exact-weight v8 supervisor."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import boto3
import runpod
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from src.glm53_user_eval.v8.artifacts import atomic_json, sha256_file
from src.glm53_user_eval.v8.on_pod import run_supervisor

MODEL_ID = "zai-org/GLM-5.3-Flash"
REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
REPO_ROOT = Path("/app/repo")
SOURCE_ROOT = Path("/app/reference/transluce-user-awareness")
EPHEMERAL_ROOT = Path("/tmp/glm53-v8")
ARTIFACT_ROOT = EPHEMERAL_ROOT / "artifacts/glm53_user_eval/v8"
INPUT_ROOT = EPHEMERAL_ROOT / "input"
PREREG = REPO_ROOT / "pipelines/glm53_user_eval/v8/configs/prereg_v8_whitebox_mechanism.yaml"
PREREG_TAG = "glm53-user-eval-v8-preregistered-v1.7"
EXPECTED_GPU_COUNTS = {3}
EXPECTED_GPU_NAME = "NVIDIA H200"
EXPECTED_HOURLY_RATE_CAP_USD = 14.50
EXPECTED_MAXIMUM_RUNTIME_HOURS = 6.0
S3_ENDPOINT = "https://s3api-us-ks-2.runpod.io/"
S3_BUCKET = "a9diryunoj"
S3_INPUT_PREFIX = "glm53-v8-input/v1.7"
S3_RESULT_PREFIX = "glm53-v8-results/v1.7"
TRANSFER = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=64 * 1024**2,
    max_concurrency=8,
    use_threads=True,
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def sanitize_embedded_checkout() -> dict[str, Any]:
    """Restore the disposable image checkout to the exact preregistered tag.

    Python/package startup can create untracked files in the baked checkout on a
    long-lived Serverless worker.  Record them, restore tracked files from the
    immutable preregistration tag, remove only untracked paths inside /app/repo,
    and fail closed unless the resulting checkout is exact and clean.
    """

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _git("status", "--porcelain=v1", "--untracked-files=all")
    head_before = _git("rev-parse", "HEAD")
    tag_commit = _git("rev-list", "-n", "1", PREREG_TAG)
    if head_before != tag_commit:
        raise RuntimeError(
            f"embedded checkout is not the preregistered commit: {head_before}"
        )

    if before:
        subprocess.check_call(
            [
                "git",
                "restore",
                "--source",
                PREREG_TAG,
                "--staged",
                "--worktree",
                "--",
                ".",
            ],
            cwd=REPO_ROOT,
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=REPO_ROOT,
        )
        for relative_raw in untracked_raw.split(b"\0"):
            if not relative_raw:
                continue
            relative = relative_raw.decode("utf-8")
            target = (REPO_ROOT / relative).resolve()
            if REPO_ROOT.resolve() not in target.parents:
                raise RuntimeError(f"untracked path escapes embedded checkout: {relative}")
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()

    after = _git("status", "--porcelain=v1", "--untracked-files=all")
    head_after = _git("rev-parse", "HEAD")
    report = {
        "schema_version": "glm53_v8_serverless_checkout_sanitization_v1",
        "prereg_tag": PREREG_TAG,
        "head_before": head_before,
        "head_after": head_after,
        "tag_commit": tag_commit,
        "dirty_before": before.splitlines(),
        "dirty_after": after.splitlines(),
        "passed": head_after == tag_commit and not after,
    }
    atomic_json(
        ARTIFACT_ROOT / "infrastructure/serverless_checkout_sanitization.json",
        report,
    )
    if report["passed"] is not True:
        raise RuntimeError(f"embedded checkout remains dirty: {after!r}")
    return report


def cached_snapshot() -> Path:
    path = (
        Path("/runpod-volume/huggingface-cache/hub")
        / "models--zai-org--GLM-5.3-Flash"
        / "snapshots"
        / REVISION
    )
    if not path.is_dir():
        raise RuntimeError(f"exact cached snapshot is absent: {path}")
    return path


def _s3_client() -> Any:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(name):
            raise RuntimeError(f"required S3 credential environment variable is absent: {name}")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name="US-KS-2",
        config=Config(
            connect_timeout=30,
            read_timeout=7200,
            max_pool_connections=16,
            retries={"max_attempts": 8, "mode": "standard"},
        ),
    )


def download_inputs() -> dict[str, str]:
    """Download and verify the frozen compact input bundle."""

    import json

    client = _s3_client()
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = INPUT_ROOT / "expected_hashes.json"
    client.download_file(
        S3_BUCKET,
        f"{S3_INPUT_PREFIX}/expected_hashes.json",
        str(manifest_path),
        Config=TRANSFER,
    )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("prereg_tag") != PREREG_TAG:
        raise RuntimeError("S3 input manifest does not name the v1.7 preregistration tag")
    files = expected.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("S3 input manifest has no files")
    observed: dict[str, str] = {}
    for relative, expected_hash in sorted(files.items()):
        target = (INPUT_ROOT / relative).resolve()
        if INPUT_ROOT.resolve() not in target.parents:
            raise RuntimeError(f"S3 input escapes input root: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        client.download_file(
            S3_BUCKET,
            f"{S3_INPUT_PREFIX}/{relative}",
            str(partial),
            Config=TRANSFER,
        )
        observed_hash = sha256_file(partial)
        if observed_hash != expected_hash:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"S3 input hash mismatch: {relative}")
        partial.replace(target)
        observed[relative] = observed_hash
    return observed


def upload_artifacts() -> dict[str, str]:
    """Upload every completed local artifact and return its hash map."""

    client = _s3_client()
    uploaded: dict[str, str] = {}
    if not ARTIFACT_ROOT.exists():
        return uploaded
    for path in sorted(item for item in ARTIFACT_ROOT.rglob("*") if item.is_file()):
        if path.name.endswith(".partial") or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(ARTIFACT_ROOT).as_posix()
        client.upload_file(
            str(path),
            S3_BUCKET,
            f"{S3_RESULT_PREFIX}/{relative}",
            Config=TRANSFER,
        )
        uploaded[relative] = sha256_file(path)
    return uploaded


class ArtifactUploader:
    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                upload_artifacts()
            except BaseException as exc:  # noqa: BLE001 - re-raised by the handler thread
                self.error = exc
                return

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.error is not None:
            raise RuntimeError("periodic S3 artifact upload failed") from self.error
        upload_artifacts()


def prepare_paths() -> Path:
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count not in EXPECTED_GPU_COUNTS:
        raise RuntimeError(f"expected one of {sorted(EXPECTED_GPU_COUNTS)} GPUs, found {gpu_count}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if any(EXPECTED_GPU_NAME not in name for name in gpu_names):
        raise RuntimeError(f"expected only {EXPECTED_GPU_NAME} GPUs, found {gpu_names}")
    model_path = EPHEMERAL_ROOT / "models/GLM-5.3-Flash" / REVISION
    model_path.parent.mkdir(parents=True, exist_ok=True)
    target = cached_snapshot()
    if model_path.is_symlink():
        if model_path.resolve() != target.resolve():
            raise RuntimeError("existing model symlink points to the wrong snapshot")
    elif model_path.exists():
        raise RuntimeError("refusing to replace an existing non-symlink model path")
    else:
        model_path.symlink_to(target, target_is_directory=True)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded = download_inputs()
    input_mapping = {
        Path("m0_decision.json"): Path("decisions/m0_decision.json"),
        Path("m1_decision.json"): Path("m1/proxy_contract.json"),
        Path("transcript_cache.jsonl"): Path("cache/v7_transcripts_25.jsonl"),
        Path("transcript_cache.manifest.json"): Path("cache/v7_transcripts_25_manifest.json"),
    }
    for source_relative, relative in input_mapping.items():
        if source_relative.as_posix() not in downloaded:
            raise RuntimeError(f"required S3 bootstrap file is absent: {source_relative}")
        source = INPUT_ROOT / source_relative
        destination = ARTIFACT_ROOT / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return model_path


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    command = payload.get("command")
    if command not in {"rate_probe", "supervise_v8"}:
        raise ValueError("accepted commands are rate_probe and supervise_v8")
    if payload.get("revision") != REVISION:
        raise ValueError("job revision does not match the preregistered checkpoint")
    if command == "rate_probe":
        if int(payload.get("hold_seconds", -1)) != 90:
            raise ValueError("rate probe must hold the worker for exactly 90 seconds")
        checkout = sanitize_embedded_checkout()
        model_path = prepare_paths()
        worker_id = os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_WORKER_ID")
        if not worker_id:
            raise RuntimeError("RunPod worker identity environment variable is absent")
        shard_count = len(list(model_path.glob("model-*-of-*.safetensors")))
        if shard_count != 62:
            raise RuntimeError(f"expected 62 model shards, found {shard_count}")
        import torch

        report = {
            "schema_version": "glm53_v8_serverless_rate_probe_v1",
            "job_id": job.get("id"),
            "model_revision": REVISION,
            "model_path": str(model_path),
            "worker_id": worker_id,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_count": torch.cuda.device_count(),
            "weight_shards": shard_count,
            "checkout": checkout,
            "hold_seconds": 90,
            "model_forward_executed": False,
            "scientific_rows": 0,
            "started_at_monotonic": time.monotonic(),
        }
        atomic_json(ARTIFACT_ROOT / "infrastructure/serverless_rate_probe.json", report)
        upload_artifacts()
        time.sleep(90)
        return report
    rate = float(payload.get("aggregate_hourly_rate_usd", -1.0))
    if rate <= 0 or rate > EXPECTED_HOURLY_RATE_CAP_USD:
        raise ValueError("job hourly rate exceeds the preregistered cap")
    if float(payload.get("maximum_runtime_hours", -1.0)) != EXPECTED_MAXIMUM_RUNTIME_HOURS:
        raise ValueError("job runtime does not match the preregistered limit")
    expected_worker_id = str(payload.get("expected_worker_id", ""))
    observed_worker_id = os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_WORKER_ID")
    if not expected_worker_id or observed_worker_id != expected_worker_id:
        raise RuntimeError("scientific job did not land on the rate-probed worker")
    sanitize_embedded_checkout()
    model_path = prepare_paths()
    invocation = {
        "schema_version": "glm53_v8_serverless_invocation_v1",
        "job_id": job.get("id"),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_path": str(model_path),
        "prereg_sha256": sha256_file(PREREG),
        "image_commit": os.environ.get("GLM53_IMAGE_COMMIT", "unknown"),
        "aggregate_hourly_rate_usd": rate,
        "maximum_runtime_hours": EXPECTED_MAXIMUM_RUNTIME_HOURS,
        "worker_id": observed_worker_id,
    }
    atomic_json(ARTIFACT_ROOT / "infrastructure/serverless_invocation.json", invocation)
    with ArtifactUploader():
        summary = run_supervisor(
            repo_root=REPO_ROOT,
            source_root=SOURCE_ROOT,
            artifact_root=ARTIFACT_ROOT,
            prereg_path=PREREG,
            full_rehash=True,
            hourly_rate_usd=rate,
        )
    response = {
        "schema_version": "glm53_v8_serverless_response_v1",
        "summary": summary,
        "artifact_root": str(ARTIFACT_ROOT),
        "supervisor_summary_sha256": sha256_file(ARTIFACT_ROOT / "supervisor_summary.json"),
    }
    atomic_json(ARTIFACT_ROOT / "infrastructure/serverless_response.json", response)
    return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
