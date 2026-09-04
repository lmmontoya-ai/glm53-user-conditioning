"""Direction construction, probe coordinate conversion, and null controls."""

from __future__ import annotations

import numpy as np


def paired_mean_direction(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[0] == 0:
        raise ValueError("paired direction inputs must be equal non-empty matrices")
    raw = np.mean(positive - negative, axis=0)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("mean direction has zero or invalid norm")
    return raw, raw / norm


def raw_logistic_weights(standardized_weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if standardized_weights.shape != scale.shape or np.any(scale <= 0):
        raise ValueError("probe weights and positive scaler values must align")
    return standardized_weights / scale


def random_directions(
    hidden_size: int,
    count: int,
    *,
    seed: int,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    if hidden_size <= 0 or count <= 0:
        raise ValueError("random direction dimensions must be positive")
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, hidden_size))
    if orthogonal_to is not None:
        direction = np.asarray(orthogonal_to, dtype=float)
        direction = direction / np.linalg.norm(direction)
        vectors -= (vectors @ direction)[:, None] * direction[None, :]
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("random direction construction produced a zero vector")
    return vectors / norms[:, None]


def residualize_direction(direction: np.ndarray, span_vectors: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    span = np.asarray(span_vectors, dtype=float)
    if span.ndim != 2 or span.shape[1] != vector.shape[0]:
        raise ValueError("span vectors must align with direction")
    basis, _ = np.linalg.qr(span.T)
    residual = vector - basis @ (basis.T @ vector)
    norm = np.linalg.norm(residual)
    if norm == 0:
        raise ValueError("direction lies entirely inside residualization span")
    return residual / norm


def natural_gap_scale(gaps: np.ndarray, alpha: float, layer_count: int) -> np.ndarray:
    if layer_count <= 0 or gaps.shape != (layer_count,):
        raise ValueError("one natural projection gap is required per layer")
    return alpha * gaps / layer_count
