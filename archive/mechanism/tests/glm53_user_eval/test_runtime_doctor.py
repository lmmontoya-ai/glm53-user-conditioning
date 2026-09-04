from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.glm53_user_eval.runtime_doctor import (
    additive_probe_hook,
    registered_hook,
    stage_model_snapshot,
    zero_delta_hook,
)


class _IdentityLayer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_zero_delta_hook_is_bit_exact() -> None:
    streams = torch.randn(2, 3, 4, 8)
    assert torch.equal(zero_delta_hook(None, None, streams), streams)


def test_registered_hook_is_removed_after_exception() -> None:
    layer = _IdentityLayer()
    with pytest.raises(RuntimeError, match="intentional"):
        with registered_hook(layer, zero_delta_hook):
            assert len(layer._forward_hooks) == 1
            raise RuntimeError("intentional")
    assert len(layer._forward_hooks) == 0


def test_additive_probe_observes_requested_delta() -> None:
    streams = torch.randn(1, 2, 4, 8)
    delta = torch.zeros(8)
    delta[0] = 0.25
    trace: dict[str, object] = {}
    changed = additive_probe_hook(delta, trace)(None, None, (streams, "routing"))
    assert changed[1] == "routing"
    assert float(trace["max_delta_error"]) <= 1e-6


def test_stage_model_snapshot_validates_and_hashes_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    destination = tmp_path / revision

    def fake_download(**kwargs: object) -> None:
        assert kwargs["revision"] == revision
        destination.mkdir()
        (destination / "model-00001-of-00002.safetensors").write_bytes(b"one")
        (destination / "model-00002-of-00002.safetensors").write_bytes(b"two")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    report = stage_model_snapshot(
        model_id="example/model",
        revision=revision,
        output_root=tmp_path,
        expected_shards=2,
        expected_bytes=6,
        max_workers=1,
    )
    assert report["safetensor_shards"] == 2
    assert report["safetensor_bytes"] == 6
    assert sorted(report["safetensor_sha256"]) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    json.dumps(report)


def test_stage_model_snapshot_fails_on_incomplete_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    revision = "b" * 40
    destination = tmp_path / revision

    def fake_download(**_kwargs: object) -> None:
        destination.mkdir()
        (destination / "model-00001-of-00002.safetensors").write_bytes(b"one")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    with pytest.raises(ValueError, match="expected 2"):
        stage_model_snapshot(
            model_id="example/model",
            revision=revision,
            output_root=tmp_path,
            expected_shards=2,
            expected_bytes=6,
            max_workers=1,
        )

