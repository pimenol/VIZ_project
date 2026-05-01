"""Load a ProteinTTT run from disk into numpy arrays."""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"step_(\d+)\.pdb$")

_REQUIRED_TSV_COLS = ("step", "loss", "plddt", "lddt")

_AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class PtttRun:
    steps: np.ndarray            # (S,) int
    loss: np.ndarray             # (S,) float, NaN for missing
    plddt_mean: np.ndarray       # (S,) float
    lddt: np.ndarray             # (S,) float
    plddt_matrix: np.ndarray     # (S, N) float32
    plddt_delta: np.ndarray      # (S, N) float32, plddt_matrix - plddt_matrix[0]
    embeddings_hd: np.ndarray    # (S, N, D) float32
    embedding_kind: str          # "esm" | "ca"
    aa_sequence: str             # length N, single-letter codes ("X" if unknown)
    ss_matrix: np.ndarray        # (S, N) uint8, 0=H helix, 1=E sheet, 2=C coil
    n_steps: int
    n_residues: int
    best_step: int
    best_plddt: float


def ss_segments(ss_row: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode an SS row → list of (start, end_inclusive, label)."""
    n = ss_row.size
    if n == 0:
        return []
    change = np.where(np.diff(ss_row) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [n - 1]))
    return [(int(s), int(e), int(ss_row[s])) for s, e in zip(starts, ends)]


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


def _parse_pdb_ca(pdb_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Per-residue pLDDT (B-factor, cols 60:66), xyz (cols 30:54), and AA letter (cols 17:20)."""
    plddts: list[float] = []
    coords: list[tuple[float, float, float]] = []
    aa: list[str] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                plddts.append(float(line[60:66]))
            except ValueError:
                log.warning("Bad B-factor in %s: %r", pdb_path.name, line.rstrip())
                plddts.append(np.nan)
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                log.warning("Bad xyz in %s: %r", pdb_path.name, line.rstrip())
                coords.append((np.nan, np.nan, np.nan))
            aa.append(_AA3_TO_AA1.get(line[17:20].strip(), "X"))
    return (
        np.array(plddts, dtype=np.float32),
        np.array(coords, dtype=np.float32),
        "".join(aa),
    )


def _load_embeddings_esm(
    embeddings_dir: Path,
    steps: np.ndarray,
    n_residues: int,
) -> np.ndarray:
    """Stack step_<i>.npy files into [S, N, D] float32. Errors if any file is missing."""
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    arrays: list[np.ndarray] = []
    for s in steps:
        path = embeddings_dir / f"step_{int(s)}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding for step {int(s)}: {path}")
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Embedding {path} has shape {arr.shape}, expected 2D (N, D)")
        if arr.shape[0] != n_residues:
            raise ValueError(
                f"Embedding {path} has {arr.shape[0]} rows, expected {n_residues}"
            )
        arrays.append(arr.astype(np.float32, copy=False))
    return np.stack(arrays, axis=0)


def _compute_ss_matrix(
    pdb_by_step: dict[int, Path],
    steps: np.ndarray,
    n_residues: int,
) -> np.ndarray:
    """For each step, run SS classification on its PDB and stack into [S, N] uint8."""
    from structure_detects import describe_protein_structure  # local import: heavy

    rows: list[np.ndarray] = []
    for s in steps:
        s_int = int(s)
        if s_int not in pdb_by_step:
            rows.append(np.full(n_residues, 2, dtype=np.uint8))  # missing -> coil
            continue
        ss = describe_protein_structure(str(pdb_by_step[s_int]))
        if ss.size != n_residues:
            raise ValueError(
                f"SS array for step {s_int} has length {ss.size}, expected {n_residues}"
            )
        rows.append(ss.astype(np.uint8, copy=False))
    return np.stack(rows, axis=0)


def _load_ss_matrix(
    pdb_by_step: dict[int, Path],
    steps: np.ndarray,
    n_residues: int,
    cache_path: Path | None,
    recompute: bool,
) -> np.ndarray:
    """Load ss_matrix from cache if shape matches, else compute and cache."""
    expected_shape = (steps.size, n_residues)
    if cache_path is not None and cache_path.exists() and not recompute:
        try:
            cached = np.load(cache_path)
            if cached.shape == expected_shape:
                log.debug("Loaded SS matrix from cache %s", cache_path)
                return cached.astype(np.uint8, copy=False)
            log.warning(
                "SS cache %s has shape %s, expected %s — recomputing",
                cache_path, cached.shape, expected_shape,
            )
        except Exception as exc:
            log.warning("Failed to read SS cache %s (%s) — recomputing", cache_path, exc)

    ss = _compute_ss_matrix(pdb_by_step, steps, n_residues)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, ss)
            log.debug("Wrote SS matrix cache to %s", cache_path)
        except Exception as exc:
            log.warning("Failed to write SS cache %s: %s", cache_path, exc)
    return ss


def _discover_pdbs(pdbs_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in pdbs_dir.iterdir():
        m = _STEP_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"No step_<i>.pdb files found in {pdbs_dir}")
    return out


def load_run(
    tsv_path: Path,
    pdbs_dir: Path,
    embedding_mode: Literal["esm", "ca"] = "esm",
    embeddings_dir: Path | None = None,
    recompute_ss: bool = False,
) -> PtttRun:
    tsv = _parse_tsv(tsv_path)
    pdb_by_step = _discover_pdbs(pdbs_dir)

    steps = tsv["steps"]
    missing = [int(s) for s in steps if int(s) not in pdb_by_step]
    if missing:
        log.warning("TSV references steps with no PDB: %s", missing)
    extra = sorted(set(pdb_by_step) - set(int(s) for s in steps))
    if extra:
        log.warning("PDB folder has extra steps not in TSV: %s", extra)

    plddt_rows: list[np.ndarray] = []
    xyz_rows: list[np.ndarray] = []
    aa_sequence = ""
    n_residues: int | None = None
    for s in steps:
        s_int = int(s)
        if s_int not in pdb_by_step:
            if n_residues is None:
                raise RuntimeError(
                    f"Step {s_int} missing PDB and no prior step to size from"
                )
            plddt_rows.append(np.full(n_residues, np.nan, dtype=np.float32))
            xyz_rows.append(np.full((n_residues, 3), np.nan, dtype=np.float32))
            continue
        plddt, xyz, aa = _parse_pdb_ca(pdb_by_step[s_int])
        if n_residues is None:
            n_residues = plddt.size
            aa_sequence = aa
        elif plddt.size != n_residues:
            raise ValueError(
                f"Step {s_int} has {plddt.size} CA atoms, expected {n_residues}"
            )
        plddt_rows.append(plddt)
        xyz_rows.append(xyz)

    plddt_matrix = np.stack(plddt_rows, axis=0)  # (S, N) float32
    plddt_delta = (plddt_matrix - plddt_matrix[0]).astype(np.float32)

    embeddings_hd, embedding_kind = _resolve_embeddings(
        embedding_mode=embedding_mode,
        embeddings_dir=embeddings_dir,
        pdbs_dir=pdbs_dir,
        steps=steps,
        n_residues=n_residues,
        xyz_rows=xyz_rows,
    )

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

    ss_matrix = _load_ss_matrix(
        pdb_by_step=pdb_by_step,
        steps=steps,
        n_residues=plddt_matrix.shape[1],
        cache_path=pdbs_dir.parent / "ss_matrix.npy",
        recompute=recompute_ss,
    )

    return PtttRun(
        steps=steps,
        loss=tsv["loss"],
        plddt_mean=plddt_mean,
        lddt=tsv["lddt"],
        plddt_matrix=plddt_matrix,
        plddt_delta=plddt_delta,
        embeddings_hd=embeddings_hd,
        embedding_kind=embedding_kind,
        aa_sequence=aa_sequence,
        ss_matrix=ss_matrix,
        n_steps=plddt_matrix.shape[0],
        n_residues=plddt_matrix.shape[1],
        best_step=best_step,
        best_plddt=best_plddt,
    )


def _resolve_embeddings(
    embedding_mode: Literal["esm", "ca"],
    embeddings_dir: Path | None,
    pdbs_dir: Path,
    steps: np.ndarray,
    n_residues: int,
    xyz_rows: list[np.ndarray],
) -> tuple[np.ndarray, str]:
    """Return (embeddings_hd, kind). ESM auto-falls-back to CA if files missing."""
    if embedding_mode == "ca":
        return np.stack(xyz_rows, axis=0), "ca"

    emb_dir = embeddings_dir if embeddings_dir is not None else pdbs_dir.parent / "embeddings"
    try:
        return _load_embeddings_esm(emb_dir, steps, n_residues), "esm"
    except FileNotFoundError as exc:
        log.warning("ESM embeddings unavailable (%s); falling back to CA coordinates", exc)
        return np.stack(xyz_rows, axis=0), "ca"
