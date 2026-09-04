from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runtime_module = importlib.import_module("src.glm53_user_eval.v11.runtime")
PREREG = ROOT / "pipelines/glm53_user_eval/v18/configs/prereg_v18_b300_execution.yaml"
RUNTIME = ROOT / "pipelines/glm53_user_eval/v18/configs/runtime_v18.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v18_binds_v17_science_without_changes() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    assert prereg["parent"]["scientific_rows"] == 0
    assert prereg["parent"]["model_forwards"] == 0
    assert prereg["infrastructure_amendment"]["scientific_change"] is False
    for record in prereg["science_lock"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_v18_is_exactly_two_b300s_and_bounded() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))
    expected = "NVIDIA B300 SXM6 AC"
    assert prereg["infrastructure_amendment"]["gpu_id"] == expected
    assert prereg["infrastructure_amendment"]["gpu_count"] == 2
    assert runtime["runpod"]["gpu_id"] == expected
    assert runtime["runpod"]["gpu_count"] == 2
    assert runtime["runtime_checks"]["expected_cuda_devices"] == 2
    assert prereg["budget"]["compute_hard_cap_usd"] == 30.0
    assert prereg["budget"]["wall_clock_hard_cap_minutes"] == 110
    assert prereg["infrastructure_amendment"]["gpu_fallback"] is False
    assert prereg["infrastructure_amendment"]["network_volume_attached"] is False


def test_v18_does_not_unlock_extra_science() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    assert prereg["execution"]["positive_control_before_user_rows"] is True
    assert prereg["execution"]["local_parity_before_causal_user_test"] is True
    assert prereg["execution"]["first_cot_allowed"] is False
    assert prereg["execution"]["generation_allowed"] is False


def test_launcher_and_bootstrap_share_transport_and_hardware_bindings() -> None:
    launcher = (ROOT / "infra/runpod/new_glm53_v17_hua_pod.ps1").read_text(
        encoding="utf-8"
    )
    bootstrap = (ROOT / "infra/runpod/bootstrap_glm53_v17.sh").read_text(
        encoding="utf-8"
    )
    for value in (
        "glm53-v18-results",
        "glm53-user-eval-v18-runtime-source-v2",
        "pipelines/glm53_user_eval/v18/configs/runtime_v18.yaml",
        "NVIDIA B300 SXM6 AC",
    ):
        assert value in launcher
        assert value in bootstrap
    assert '$ExpectedGpuCount = 2' in launcher
    assert "torch.cuda.device_count() != 2" in bootstrap
    assert "immutable_v17_runtime.yaml" in launcher
    assert "immutable_v17_runtime.yaml" in bootstrap
    assert "immutable_v18_runtime.yaml" in launcher
    assert "immutable_v18_runtime.yaml" in bootstrap


def test_exact_transformers_source_archive_is_valid_revision_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "transformers_exact.tar.gz"
    source.write_bytes(b"exact transported source")

    class Distribution:
        @staticmethod
        def read_text(name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps({"url": source.as_uri()})

    monkeypatch.setattr(runtime_module, "installed_vcs_commit", lambda _name: None)
    monkeypatch.setattr(
        runtime_module.importlib.metadata, "distribution", lambda _name: Distribution()
    )
    config = {
        "software": {
            "transformers_commit": "a" * 40,
            "transformers_source_filename": source.name,
            "transformers_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    }
    assert runtime_module.verify_transformers_source(config) == "a" * 40
