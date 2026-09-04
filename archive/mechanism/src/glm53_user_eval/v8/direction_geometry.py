"""Direction geometry reported only after causal decisions freeze."""

from __future__ import annotations

import numpy as np


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        raise ValueError("cosine requires nonzero vectors")
    return float(left @ right / denom)


def residualize(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("basis must be [directions, hidden]")
    q, _ = np.linalg.qr(matrix.T)
    result = np.asarray(vector, dtype=np.float64) - q @ (q.T @ vector)
    return result


def whitened_cosine(
    left: np.ndarray, right: np.ndarray, covariance: np.ndarray, ridge: float = 1e-5
) -> float:
    values, vectors = np.linalg.eigh(
        np.asarray(covariance, dtype=np.float64) + ridge * np.eye(len(left))
    )
    whitening = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return cosine(whitening @ left, whitening @ right)
