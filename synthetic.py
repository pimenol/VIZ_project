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
    perplexity = np.exp(np.where(np.isnan(loss), 0.0, loss))
    perplexity[0] = np.nan

    # tm_score and lddt: rise then dip slightly.
    tm = 0.55 + 0.30 * rise * decline + rng.normal(0.0, 0.01, size=n_steps)
    lddt = 0.50 + 0.35 * rise * decline + rng.normal(0.0, 0.01, size=n_steps)

    best_step = int(np.argmax(plddt_mean))
    best_plddt = float(plddt_mean[best_step])

    return PtttRun(
        steps=steps,
        loss=loss,
        perplexity=perplexity,
        plddt_mean=plddt_mean,
        tm_score=tm.astype(np.float64),
        lddt=lddt.astype(np.float64),
        plddt_matrix=plddt,
        plddt_delta=plddt_delta,
        n_steps=n_steps,
        n_residues=n_residues,
        best_step=best_step,
        best_plddt=best_plddt,
    )
