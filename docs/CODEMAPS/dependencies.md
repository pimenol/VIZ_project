<!-- Generated: 2026-05-01 | Files scanned: 16 -->

# Dependencies

No package manifest yet (no `pyproject.toml` / `requirements.txt`). Deps below are inferred from imports.

## Runtime — required

| Package | Used by | Purpose |
|---|---|---|
| `PySide6` | every view, `main.py` | Qt6 widgets, signals, QGraphics, QImage, QOpenGLWidget |
| `numpy` | everything except chart_axes | All numeric arrays, vectorized ARGB packing, SVD-based PCA |

Python 3.10+ required (uses `int \| None`, `list[T]` runtime annotations directly, no `from __future__ import annotations`).

## Runtime — lazy / optional

Imported inside the function that needs them so the rest of the app runs without them:

| Package | Imported in | Triggered when |
|---|---|---|
| `umap-learn` | [reduction.py:36](reduction.py#L36) | `--reduction umap` |
| `scikit-learn` (`sklearn.manifold.TSNE`) | [reduction.py:47](reduction.py#L47) | `--reduction tsne` |

UMAP transitively pulls `numba` + `scipy`. Cache misses only — successful runs hit `cache_2d_<method>.npy`.

## Hard rule

Forbidden imports (per project spec): `matplotlib`, `seaborn`, `plotly`, `bokeh`, `pyqtgraph`, `vispy`. All visualization is custom-rendered with QGraphics primitives + `QImage`.

## External services

None. App is fully offline; reads local TSV/PDB/.npy files and writes caches next to the inputs.

## Internal-only modules (no external deps)

- [chart_axes.py](chart_axes.py) — pure-Qt
- [colors.py](colors.py) — Qt + numpy
- [controller.py](controller.py) — `QObject` only
- [points_item.py](points_item.py) — Qt + numpy
- [structure_detects.py](structure_detects.py) — pure numpy (despite the name, no biopython/biotite)

## Suggested manifest (not yet checked in)

```
PySide6>=6.5
numpy>=1.24
umap-learn>=0.5      # optional
scikit-learn>=1.3    # optional
```
