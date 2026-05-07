import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReductionResult:
    coords_2d: np.ndarray
    method: str
    explained_variance_ratio: tuple[float, float] | None


def reduce_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"reduce_pca expects 2D array, got shape {X.shape}")
    M, D = X.shape
    if n_components > min(M, D):
        raise ValueError(f"n_components={n_components} exceeds min(M, D)={min(M, D)}")

    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :n_components] * s[:n_components]
    eigvals = (s ** 2) / max(M - 1, 1)
    total_var = float(eigvals.sum())
    ratio = (eigvals[:n_components] / total_var) if total_var > 0 else np.zeros(n_components, dtype=float)
    return Y.astype(np.float32, copy=False), ratio.astype(float, copy=False)


def reduce_joint(
    embeddings: np.ndarray,
    method: str,
    cache_dir: Path | None,
    cache_key: str,
    recompute: bool = False,
) -> ReductionResult:
    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be [S, N, D], got shape {embeddings.shape}")
    S, N, D = embeddings.shape
    method = method.lower()
    if method != "pca":
        raise ValueError(f"unknown method {method!r}")

    if cache_dir is not None and not recompute:
        cached = _load_cache(cache_dir, method, cache_key, expected_shape=(S, N, 2))
        if cached is not None:
            return cached

    flat = embeddings.reshape(S * N, D).astype(np.float32, copy=False)
    Y, var_ratio = reduce_pca(flat, n_components=2)
    result = ReductionResult(
        coords_2d=Y.reshape(S, N, 2),
        method="pca",
        explained_variance_ratio=(float(var_ratio[0]), float(var_ratio[1])),
    )

    if cache_dir is not None:
        _save_cache(cache_dir, method, cache_key, result)

    return result


def _cache_paths(cache_dir: Path, method: str, cache_key: str) -> tuple[Path, Path]:
    stem = f"cache_2d_{method}__{cache_key}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _load_cache(
    cache_dir: Path,
    method: str,
    cache_key: str,
    expected_shape: tuple[int, int, int],
) -> ReductionResult | None:
    npy_path, meta_path = _cache_paths(cache_dir, method, cache_key)
    if not npy_path.exists() or not meta_path.exists():
        return None
    try:
        coords = np.load(npy_path)
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if tuple(coords.shape) != expected_shape:
        return None
    var_ratio = meta.get("explained_variance_ratio")
    return ReductionResult(
        coords_2d=coords.astype(np.float32, copy=False),
        method=meta.get("method", method),
        explained_variance_ratio=tuple(var_ratio) if var_ratio is not None else None,
    )


def _save_cache(cache_dir: Path, method: str, cache_key: str, result: ReductionResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path, meta_path = _cache_paths(cache_dir, method, cache_key)
    np.save(npy_path, result.coords_2d)
    meta = {
        "method": result.method,
        "shape": list(result.coords_2d.shape),
        "explained_variance_ratio": list(result.explained_variance_ratio) if result.explained_variance_ratio else None,
        "cache_key": cache_key,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
