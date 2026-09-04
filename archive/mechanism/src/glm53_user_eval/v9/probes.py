"""Paper-contract sequence and token-MIL probes for v9."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


@dataclass(frozen=True)
class LinearProbe:
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    threshold: float
    best_epoch: int
    val_auroc: float
    val_brier: float


def _labels_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in rows],
        dtype=np.int64,
    )


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))
    fpr, tpr, _ = roc_curve(y, score)
    eligible = fpr[tpr >= 0.8]
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "fpr_at_80_tpr": float(eligible.min()) if len(eligible) else 1.0,
    }


def _balanced_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.concatenate(([-np.inf], np.unique(score), [np.inf]))
    best = (-np.inf, 0.0)
    for threshold in candidates:
        predicted = score >= threshold
        tpr = float(predicted[y == 1].mean())
        tnr = float((~predicted[y == 0]).mean())
        value = 0.5 * (tpr + tnr)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def _adamw_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    torch.manual_seed(seed)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    vx = torch.as_tensor(val_x, dtype=torch.float32)
    vy = torch.as_tensor(val_y, dtype=torch.float32)
    model = torch.nn.Linear(train_x.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    remaining = patience
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for start in range(0, len(x), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x[index]).squeeze(1), y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(vx).squeeze(1), vy))
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("AdamW probe produced no checkpoint")
    model.load_state_dict(best_state)
    return (
        model.weight.detach().numpy()[0].astype(np.float32),
        float(model.bias.detach().numpy()[0]),
        best_epoch,
    )


def fit_sequence_layers(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    labels: np.ndarray | None = None,
    config: dict[str, Any],
    seed: int,
) -> tuple[LinearProbe, list[dict[str, Any]]]:
    if features.ndim != 3 or features.shape[1:] != (45, 4096):
        raise ValueError(f"expected [rows,45,4096], got {features.shape}")
    y = np.asarray(labels, dtype=np.int64) if labels is not None else _labels_from_rows(rows)
    splits = np.asarray([row["split"] for row in rows])
    train = splits == "train"
    val = splits == "val"
    if not np.isin(y[train | val], [0, 1]).all():
        raise ValueError("training and validation rows must have binary labels")
    candidates: list[LinearProbe] = []
    layer_rows: list[dict[str, Any]] = []
    for layer in range(45):
        train_x = features[train, layer].astype(np.float32)
        val_x = features[val, layer].astype(np.float32)
        mean = train_x.mean(0)
        scale = train_x.std(0)
        scale[scale == 0] = 1
        weight, bias, best_epoch = _adamw_linear(
            (train_x - mean) / scale,
            y[train],
            (val_x - mean) / scale,
            y[val],
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            patience=int(config["early_stopping_patience"]),
            seed=seed + layer,
        )
        val_score = ((val_x - mean) / scale) @ weight + bias
        val_metrics = _metrics(y[val], val_score)
        threshold = _balanced_threshold(y[val], val_score)
        probe = LinearProbe(
            layer=layer,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            weight=weight,
            bias=bias,
            threshold=threshold,
            best_epoch=best_epoch,
            val_auroc=val_metrics["auroc"],
            val_brier=val_metrics["brier"],
        )
        candidates.append(probe)
        layer_rows.append({"layer": layer, "best_epoch": best_epoch} | val_metrics)
    selected = min(candidates, key=lambda item: (-item.val_auroc, item.val_brier, item.layer))
    return selected, layer_rows


def score_sequence(probe: LinearProbe, features: np.ndarray) -> np.ndarray:
    return ((features[:, probe.layer] - probe.mean) / probe.scale) @ probe.weight + probe.bias


def pair_preserving_labels(
    rows: list[dict[str, Any]], *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray([row["label"] if row["label"] is not None else -1 for row in rows])
    by_pair: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["label"] is not None:
            by_pair.setdefault(str(row["pair_id"]), []).append(index)
    for indices in by_pair.values():
        pair_labels = sorted(int(labels[index]) for index in indices)
        if pair_labels != [0, 1]:
            raise ValueError("pair-preserving null requires one eval and one deploy row per pair")
        if int(rng.integers(0, 2)):
            labels[indices] = 1 - labels[indices]
    return labels


def evaluate_probe(
    probe: LinearProbe, features: np.ndarray, rows: list[dict[str, Any]], labels: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    y = np.asarray(
        labels if labels is not None else [row["label"] if row["label"] is not None else -1 for row in rows]
    )
    splits = np.asarray([row["split"] for row in rows])
    score = score_sequence(probe, features)
    output: dict[str, dict[str, float]] = {}
    for split_name in ("test", "holdout_realism", "holdout_hardneg", "holdout_mats_aux"):
        split_mask = splits == split_name
        binary = split_mask & np.isin(y, [0, 1])
        output[split_name] = _metrics(y[binary], score[binary])
        neutral = split_mask & (y == -1)
        if neutral.any():
            output[split_name]["neutral_false_positive_rate"] = float(
                np.mean(score[neutral] >= probe.threshold)
            )
            output[split_name]["neutral_mean_score"] = float(np.mean(score[neutral]))
            output[split_name]["neutral_count"] = int(neutral.sum())
    return output


def run_full_pipeline_null(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for repetition in range(reps):
        shuffled = pair_preserving_labels(rows, seed=seed + repetition)
        probe, _ = fit_sequence_layers(
            features,
            rows,
            labels=shuffled,
            config=config,
            seed=seed + 10_000 + repetition * 100,
        )
        metrics = evaluate_probe(probe, features, rows, labels=shuffled)
        outputs.append(
            {
                "repetition": repetition,
                "selected_layer": probe.layer,
                "test_auroc": metrics["test"]["auroc"],
            }
        )
    return outputs


def load_bags(feature_root: Path, rows: list[dict[str, Any]], layer: int) -> list[np.ndarray]:
    bags: list[np.ndarray] = []
    for index, row in enumerate(rows):
        part = feature_root / row["part"]
        archive = np.load(part)
        bag = archive[f"bag_l{layer:02d}"].astype(np.float32)
        if bag.ndim != 2 or bag.shape[1] != 4096 or len(bag) == 0:
            raise ValueError(f"invalid token bag for {row['sample_id']}")
        bags.append(bag)
    return bags


def fit_token_mil_layer(
    bags: list[np.ndarray],
    rows: list[dict[str, Any]],
    *,
    layer: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    split = np.asarray([row["split"] for row in rows])
    y = np.asarray([row["label"] if row["label"] is not None else -1 for row in rows])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    token_stack = np.vstack([bags[index] for index in train_indices])
    mean = token_stack.mean(0).astype(np.float32)
    scale = token_stack.std(0).astype(np.float32)
    scale[scale == 0] = 1
    standardized = [torch.as_tensor((bag - mean) / scale, dtype=torch.float32) for bag in bags]
    torch.manual_seed(seed)
    model = torch.nn.Linear(4096, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    remaining = int(config["early_stopping_patience"])
    top_k = int(config["top_k_tokens"])

    def bag_logit(index: int) -> torch.Tensor:
        token_logits = model(standardized[index]).squeeze(1)
        return torch.topk(token_logits, k=min(top_k, len(token_logits))).values.mean()

    for epoch in range(int(config["epochs"])):
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            logits = torch.stack([bag_logit(int(index)) for index in indices])
            targets = torch.as_tensor(y[indices], dtype=torch.float32)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = torch.stack([bag_logit(int(index)) for index in val_indices])
            val_loss = float(loss_fn(logits, torch.as_tensor(y[val_indices], dtype=torch.float32)))
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            remaining = int(config["early_stopping_patience"])
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("token MIL produced no checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        scores = np.asarray([float(bag_logit(index)) for index in range(len(rows))])
    threshold = _balanced_threshold(y[val_indices], scores[val_indices])
    return {
        "layer": layer,
        "best_epoch": best_epoch,
        "mean": mean,
        "scale": scale,
        "weight": model.weight.detach().numpy()[0],
        "bias": float(model.bias.detach().numpy()[0]),
        "validation": _metrics(y[val_indices], scores[val_indices]),
        "threshold": threshold,
        "scores": scores,
    }
