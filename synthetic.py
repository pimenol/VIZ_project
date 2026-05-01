"""Generate a synthetic PtttRun for development and demo."""

from __future__ import annotations

import numpy as np

from data import PtttRun


def make_demo_run(
    n_steps: int = 30,
    n_residues: int = 200,
    peak_step: int = 15,
    seed: int = 0,
) -> PtttRun:
    rng = np.random.default_rng(seed)

    steps = np.arange(n_steps, dtype=np.int32)

    # Per-residue pLDDT: rises toward peak_step, then mild decline (overfitting tail).
    s = steps.astype(np.float32)
    rise = 1.0 - np.exp(-s / max(peak_step / 2.0, 1.0))           # 0 → ~1
    decline = np.exp(-np.clip(s - peak_step, 0, None) / 30.0)      # 1 → <1 after peak
    base_curve = 50.0 + 35.0 * rise * decline                      # mean trajectory

    # Per-residue offset (some residues never confident, others always).
    res_bias = rng.normal(0.0, 8.0, size=n_residues).astype(np.float32)
    plddt = (
        base_curve[:, None]
        + res_bias[None, :]
        + rng.normal(0.0, 3.0, size=(n_steps, n_residues)).astype(np.float32)
    )
    plddt = np.clip(plddt, 0.0, 100.0).astype(np.float32)

    plddt_mean = plddt.mean(axis=1).astype(np.float64)
    plddt_delta = (plddt - plddt[0]).astype(np.float32)

    # Loss: exponential decay with noise; step 0 has no loss (real data convention).
    loss_clean = 2.5 * np.exp(-s / 12.0) + 0.3
    loss = loss_clean + rng.normal(0.0, 0.05, size=n_steps)
    loss[0] = np.nan

    lddt = 0.50 + 0.35 * rise * decline + rng.normal(0.0, 0.01, size=n_steps)

    embeddings_hd = _make_demo_embeddings(rng, n_steps, n_residues, dim=64)

    ss_matrix = _make_demo_ss(rng, n_steps, n_residues)

    best_step = int(np.argmax(plddt_mean))
    best_plddt = float(plddt_mean[best_step])

    return PtttRun(
        steps=steps,
        loss=loss,
        plddt_mean=plddt_mean,
        lddt=lddt.astype(np.float64),
        plddt_matrix=plddt,
        plddt_delta=plddt_delta,
        embeddings_hd=embeddings_hd,
        embedding_kind="esm",
        aa_sequence="X" * n_residues,
        ss_matrix=ss_matrix,
        n_steps=n_steps,
        n_residues=n_residues,
        best_step=best_step,
        best_plddt=best_plddt,
    )


def _make_demo_ss(
    rng: np.random.Generator,
    n_steps: int,
    n_residues: int,
) -> np.ndarray:
    """Per-residue SS labels with a few helix/sheet blocks; small per-step jitter at boundaries."""
    base = np.full(n_residues, 2, dtype=np.uint8)  # all coil

    pos = 0
    while pos < n_residues:
        gap = rng.integers(2, 8)
        pos += int(gap)
        if pos >= n_residues:
            break
        run = int(rng.integers(6, 18))
        end = min(pos + run, n_residues)
        label = np.uint8(rng.integers(0, 2))  # 0=H or 1=E
        base[pos:end] = label
        pos = end

    out = np.broadcast_to(base, (n_steps, n_residues)).copy()
    # Jitter ±1 residue on boundaries each step so the dock SS-evolution strip shows variety.
    boundaries = np.where(np.diff(base) != 0)[0]
    for t in range(1, n_steps):
        flip = rng.choice(boundaries, size=min(2, boundaries.size), replace=False) if boundaries.size else []
        for b in flip:
            j = b + (1 if rng.random() < 0.5 else 0)
            if 0 <= j < n_residues:
                neighbor = base[max(0, j - 1)]
                out[t, j] = neighbor
    return out


def _make_demo_embeddings(
    rng: np.random.Generator,
    n_steps: int,
    n_residues: int,
    dim: int,
) -> np.ndarray:
    """Per-residue identity + per-step drift along a fixed 2D plane in D-dim space."""
    a = rng.standard_normal(dim).astype(np.float32)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(dim).astype(np.float32)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)

    base = rng.standard_normal((n_residues, dim)).astype(np.float32) * 0.3
    alpha = rng.uniform(-1.0, 1.0, n_residues).astype(np.float32)
    beta = rng.uniform(-1.0, 1.0, n_residues).astype(np.float32)

    out = np.empty((n_steps, n_residues, dim), dtype=np.float32)
    for t in range(n_steps):
        drift = (t * alpha[:, None]) * a + (t * beta[:, None]) * b
        out[t] = base + drift + 0.05 * rng.standard_normal((n_residues, dim)).astype(np.float32)
    return out
