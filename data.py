"""Load a ProteinTTT run from disk into numpy arrays."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"step_(\d+)\.pdb$")

_REQUIRED_TSV_COLS = ("step", "loss", "plddt", "lddt")


@dataclass(frozen=True)
class PtttRun:
    steps: np.ndarray            # (S,) int
    loss: np.ndarray             # (S,) float, NaN for missing
    plddt_mean: np.ndarray       # (S,) float
    lddt: np.ndarray             # (S,) float
    plddt_matrix: np.ndarray     # (S, N) float32
    plddt_delta: np.ndarray      # (S, N) float32, plddt_matrix - plddt_matrix[0]
    n_steps: int
    n_residues: int
    best_step: int
    best_plddt: float


def _to_float(cell: str) -> float:
    cell = cell.strip()
    if not cell:
        return np.nan
    return float(cell)


def _parse_tsv(tsv_path: Path) -> dict[str, np.ndarray]:
    with tsv_path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for col in _REQUIRED_TSV_COLS:
            if col not in idx:
                raise ValueError(f"TSV {tsv_path} missing required column {col!r}")
        rows = [row for row in reader if row]

    n = len(rows)
    out: dict[str, np.ndarray] = {
        "steps": np.empty(n, dtype=np.int32),
        "loss": np.empty(n, dtype=np.float64),
        "plddt_mean": np.empty(n, dtype=np.float64),
        "lddt": np.empty(n, dtype=np.float64),
    }
    for i, row in enumerate(rows):
        out["steps"][i] = int(row[idx["step"]])
        out["loss"][i] = _to_float(row[idx["loss"]])
        out["plddt_mean"][i] = _to_float(row[idx["plddt"]])
        out["lddt"][i] = _to_float(row[idx["lddt"]])
    return out


def _parse_pdb_ca_plddt(pdb_path: Path) -> np.ndarray:
    """Extract per-residue pLDDT from CA-atom B-factors."""
    values: list[float] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                values.append(float(line[60:66]))
            except ValueError:
                log.warning("Bad B-factor in %s: %r", pdb_path.name, line.rstrip())
                values.append(np.nan)
    return np.array(values, dtype=np.float32)


def _discover_pdbs(pdbs_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in pdbs_dir.iterdir():
        m = _STEP_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"No step_<i>.pdb files found in {pdbs_dir}")
    return out


def load_run(tsv_path: Path, pdbs_dir: Path) -> PtttRun:
    tsv = _parse_tsv(tsv_path)
    pdb_by_step = _discover_pdbs(pdbs_dir)

    steps = tsv["steps"]
    missing = [int(s) for s in steps if int(s) not in pdb_by_step]
    if missing:
        log.warning("TSV references steps with no PDB: %s", missing)
    extra = sorted(set(pdb_by_step) - set(int(s) for s in steps))
    if extra:
        log.warning("PDB folder has extra steps not in TSV: %s", extra)

    rows: list[np.ndarray] = []
    n_residues: int | None = None
    for s in steps:
        s_int = int(s)
        if s_int not in pdb_by_step:
            if n_residues is None:
                raise RuntimeError(
                    f"Step {s_int} missing PDB and no prior step to size from"
                )
            rows.append(np.full(n_residues, np.nan, dtype=np.float32))
            continue
        per_res = _parse_pdb_ca_plddt(pdb_by_step[s_int])
        if n_residues is None:
            n_residues = per_res.size
        elif per_res.size != n_residues:
            raise ValueError(
                f"Step {s_int} has {per_res.size} CA atoms, expected {n_residues}"
            )
        rows.append(per_res)

    plddt_matrix = np.stack(rows, axis=0)  # (S, N) float32
    plddt_delta = (plddt_matrix - plddt_matrix[0]).astype(np.float32)

    plddt_mean = tsv["plddt_mean"]
    pdb_means = np.nanmean(plddt_matrix, axis=1)
    max_diff = float(np.nanmax(np.abs(pdb_means - plddt_mean)))
    if max_diff > 10.0:
        log.warning(
            "TSV plddt vs PDB-derived mean disagree by >10 at some steps "
            "(max diff = %.3f); B-factor column may not be pLDDT", max_diff,
        )
    else:
        log.debug("TSV vs PDB plddt max diff = %.3f", max_diff)

    if np.all(np.isnan(plddt_mean)):
        raise ValueError("All plddt_mean values are NaN; cannot determine best step")
    best_step = int(np.nanargmax(plddt_mean))
    best_plddt = float(plddt_mean[best_step])

    return PtttRun(
        steps=steps,
        loss=tsv["loss"],
        plddt_mean=plddt_mean,
        lddt=tsv["lddt"],
        plddt_matrix=plddt_matrix,
        plddt_delta=plddt_delta,
        n_steps=plddt_matrix.shape[0],
        n_residues=plddt_matrix.shape[1],
        best_step=best_step,
        best_plddt=best_plddt,
    )
