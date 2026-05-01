<!-- Generated: 2026-05-01 | Files scanned: 16 | LOC: ~4063 -->

# Architecture

Single-process PySide6 desktop app for visualizing ProteinTTT optimization runs (per-step pLDDT + ESM embeddings + secondary structure across S optimization steps × N residues).

## High-level flow

```
CLI args ──► load_run() ──► PtttRun (frozen dataclass)
                              │
                              ▼
                   SelectionController (Qt signals)
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
        4 main views (2x2)            ResidueDetailDock (right)
        T1 LineChart   T4 Embedding   ├── pLDDT trajectory
        T2 Heatmap     T3 Profile     ├── SS evolution strip
                                      ├── Embedding trajectory
                                      └── Sequence context
```

## Entry point

[main.py](main.py) — `MainWindow` builds:
- toolbar (Load/Demo, step slider+spin, color combo, residue range, SS filter checkboxes, Save PNG)
- 2×2 `QSplitter` (T1|T4 / T2|T3)
- right-docked `ResidueDetailDock` (hidden until a residue is selected)

CLI: `--demo` | `--tsv FILE --pdbs DIR` (`--embeddings esm|ca`, `--embeddings-dir DIR`, `--reduction pca|umap|tsne`, `--recompute`, `--recompute-ss`)

## Selection controller

[controller.py](controller.py) `SelectionController(QObject)` — single source of truth for cross-view state. Views emit on user input; views subscribe to update state.

Signals:
- `currentStepChanged(int)`
- `residueHoveredChanged(int)` / `residueSelectedChanged(int)` (-1 = none)
- `comparisonStepsChanged(list[int])`
- `ssClassFilterChanged(set)` (subset of {0,1,2})

## Module boundaries

| Module | Role |
|---|---|
| [data.py](data.py) | TSV/PDB/embedding loaders → `PtttRun` |
| [synthetic.py](synthetic.py) | `make_demo_run()` for `--demo` mode |
| [reduction.py](reduction.py) | PCA/UMAP/t-SNE joint fit on `[S*N, D]` + disk cache |
| [structure_detects.py](structure_detects.py) | `describe_protein_structure(pdb)` → SS labels |
| [colors.py](colors.py) | AlphaFold band colors, SS colors, vectorized ARGB array helpers |
| [chart_axes.py](chart_axes.py) | `draw_axes`, `nice_ticks` shared by all chart views |
| [points_item.py](points_item.py) | `PointsItem(QGraphicsItem)` — many points, one paint() |
| [views/](views/) | T1–T6 visualization widgets |

## Performance pattern

- Single `QGraphicsScene` per view; reused `QImage` (heatmap) and `PointsItem` (embedding) instead of one item per residue.
- Numpy-vectorized color packing (`alphafold_color_array`, `ss_color_array`) → ARGB uint32 arrays consumed by `PointsItem` and `QImage` buffers in one pass.
- `4× MSAA` viewport on chart views; offscreen render fallback when QOpenGLWidget unavailable.
- SS class filter implemented as overlay-rect fade (T2/T3) and `set_alpha_mask` (T4) — no path/image rebuilds.
