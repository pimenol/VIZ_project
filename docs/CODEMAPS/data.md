<!-- Generated: 2026-05-01 | Files scanned: 3 | LOC: ~447 -->

# Data Model

In-memory only — no database. All run data is loaded once into a frozen dataclass and shared by reference.

## PtttRun ([data.py](data.py))

```python
@dataclass(frozen=True)
class PtttRun:
    steps:         np.ndarray  # (S,)        int
    loss:          np.ndarray  # (S,)        float, NaN missing
    plddt_mean:    np.ndarray  # (S,)        float
    lddt:          np.ndarray  # (S,)        float
    plddt_matrix:  np.ndarray  # (S, N)      float32
    plddt_delta:   np.ndarray  # (S, N)      float32   = matrix - matrix[0]
    embeddings_hd: np.ndarray  # (S, N, D)   float32   D=1024 ESM, 3 CA-fallback
    embedding_kind: str        # "esm" | "ca"
    aa_sequence:   str         # length N, single-letter ("X" if unknown)
    ss_matrix:     np.ndarray  # (S, N)      uint8     0=H 1=E 2=C
    n_steps:       int
    n_residues:    int
    best_step:     int
    best_plddt:    float
```

## Loading pipeline ([data.py](data.py) `load_run`)

```
TSV ─► _parse_tsv ──► steps, loss, plddt_mean, lddt           (authoritative S)
PDBs ─► _discover_pdbs ──► {step → path}
        _parse_pdb_ca ──► plddt_matrix, CA xyz, AA letters    (per step, len N)
ESM ──► _load_embeddings_esm ──► embeddings_hd[S, N, D]       (default)
   └── fallback if .npy missing ──► CA xyz [S, N, 3]
SS ───► _load_ss_matrix ──► (S, N) uint8                      (cached on disk)
```

## Disk caches

| Path | Shape | Source-of-truth |
|---|---|---|
| `{embeddings_dir}/cache_2d_<method>.npy` + `.json` | [S, N, 2] float32 | `reduction.reduce_joint` |
| `{pdbs_dir.parent}/ss_matrix.npy` | (S, N) uint8 | `structure_detects.describe_protein_structure` per PDB |

Cache invalidation: shape mismatch → recompute. CLI `--recompute` (embeddings 2D) and `--recompute-ss` force.

## Helpers

- `ss_segments(ss_row) -> [(lo, hi, label), ...]` — RLE over a single SS row, used by all SS-track widgets.
- `_resolve_embeddings(...)` — picks dir, falls back from ESM to CA with warning.

## On-disk inputs (real data)

```
data/logs/<protein_id>/
├── <protein_id>_log.tsv       # per-step metrics
├── pdbs/step_<i>.pdb          # 0..S-1
├── embeddings/step_<i>.npy    # [N, D] float32 (optional → CA fallback)
└── ss_matrix.npy              # generated cache
```

## Synthetic demo ([synthetic.py](synthetic.py))

`make_demo_run()` produces a fully-populated `PtttRun` (S≈21, N≈80, D=64) with block-structured SS, drift-along-basis embeddings, and an improving pLDDT trajectory. Lets the GUI run without files.
