# ProteinTTT Visualization

> Interactive desktop tool for exploring how a protein language model's structure prediction evolves across **test-time-training (TTT)** optimization steps — built from scratch in PySide6, with no charting libraries.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-41CD52?logo=qt&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy&logoColor=white)
![Domain](https://img.shields.io/badge/domain-structural%20bioinformatics-8a2be2)

---

## Overview

Test-time training fine-tunes a protein language model on a single sequence at inference time. After each optimization step, the model re-predicts the 3D structure and produces a per-residue confidence score (pLDDT). This tool answers the question every researcher running TTT actually asks:

> *Did the structure get better — and **where** did it improve?*

The app loads a TTT run (one TSV of metrics + a folder of per-step PDB files) and surfaces the answer through six linked views, all responding live to a single shared selection.

## Highlights

- **Six coordinated views** — line chart, heatmap, profile overlay, embedding scatter, secondary-structure strip, and a per-residue detail dock — bound through a single `SelectionController` (brushing & linking).
- **No plotting libraries.** Every axis tick, curve, and heatmap pixel is composed from raw Qt primitives (`QPainterPath`, `QGraphicsLineItem`, `QImage`).
- **Vectorized rendering.** The full *(steps × residues)* pLDDT grid is produced in a single NumPy pass and blitted as one `QImage` — color-mode toggles refill the buffer without rebuilding the scene graph.
- **Hand-written PDB parser.** Reads CA-atom B-factors directly from fixed PDB columns; no Biopython dependency.
- **OpenGL-accelerated scatter** for the embedding view, with hover tooltips and animated transitions across steps.
- **Demo mode** generates 30 synthetic steps × 200 residues so the app is fully exercisable without any data files.

## Quick start

```bash
pip install PySide6 numpy

# Real data (one protein run)
python main.py --tsv data/logs/A0A6J5N0Y1_log.tsv --pdbs data/logs/A0A6J5N0Y1_pdbs/

# Synthetic demo (30 steps × 200 residues, no data files needed)
python main.py --demo
```

<!-- AUTO-GENERATED: CLI reference extracted from main.py _parse_args() -->
### CLI reference

| Argument | Required | Description |
|----------|----------|-------------|
| `--demo` | one of `--demo`/`--tsv` | Load 30-step × 200-residue synthetic data; no files needed |
| `--tsv FILE` | with `--pdbs` | Path to the per-step metrics TSV file |
| `--pdbs DIR` | with `--tsv` | Path to folder containing `step_<i>.pdb` files |

`--demo` and `--tsv`/`--pdbs` are mutually exclusive; exactly one mode must be specified.
<!-- END AUTO-GENERATED -->

## Views

| View | Role |
|------|------|
| **T1 — Line chart** | Mean pLDDT and lDDT across steps; peak step marked in red. |
| **T2 — Heatmap** | Per-residue pLDDT across all steps; AlphaFold bands or delta-from-step-0. |
| **T3 — Profile overlay** | Per-residue pLDDT polylines for an arbitrary set of comparison steps. |
| **T4 — Embedding scatter** | 2D projection of per-residue embeddings, animated across steps (OpenGL). |
| **T5 — Secondary-structure strip** | Per-residue SS annotation, X-axis-aligned with the heatmap and profile. |
| **T6 — Residue detail dock** | For a focused residue: pLDDT trajectory, embedding trajectory, sequence context. |

All views are linked: clicking a heatmap cell, dragging the line chart, or shift-clicking a step row updates the *step* and *residue* selections globally — every other view reacts.

## Interactions

| Action | Effect |
|--------|--------|
| Click line chart | Set current step |
| Click heatmap cell | Set current step + selected residue |
| Shift-click heatmap row | Add/remove step from comparison set |
| Check/uncheck step picker | Add/remove step from comparison set |
| Hover anywhere | Tooltip with values at cursor |
| Scroll wheel | Zoom (X-only on line chart, both axes on heatmap) |
| Drag | Pan |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| ←/→ | Previous/next step |
| Shift+←/→ | Jump 10 steps |
| Home | Step 0 |
| End | Best pLDDT step |
| +/− | Zoom in/out (heatmap) |
| R | Reset zoom |
| F | Fit to view |

### Toolbar

- **Load…** — pick a metrics TSV then a PDB folder
- **Demo** — reload synthetic data
- **Step slider/spinbox** — navigate steps
- **Color** — toggle heatmap between AlphaFold bands and delta-from-step-0
- **Res:** — filter displayed residue range
- **Save PNG…** — export current heatmap scene to PNG

## Architecture

```
main.py            MainWindow, toolbar, CLI
data.py            PtttRun dataclass + load_run()
controller.py      SelectionController (single source of truth for shared signals)
colors.py          Vectorized AlphaFold / delta palettes, SS colors
chart_axes.py      nice_ticks() + draw_axes() helpers
synthetic.py       make_demo_run() — synthetic TTT trajectories
structure_detects.py  Secondary-structure detection from PDB geometry
points_item.py     Custom QGraphicsItem for the embedding scatter
reduction.py       Dimensionality reduction for embedding view
views/
  line_chart.py    T1 — multi-panel line chart
  heatmap.py       T2 — QImage-backed heatmap
  profile_view.py  T3 — overlaid profiles + step picker
  embedding_view.py  T4 — OpenGL-accelerated 2D scatter
  ss_track.py      T5 — secondary-structure annotation strip
  residue_detail.py  T6 — per-residue detail dock
```

**Design principle:** views never talk to each other. They emit and listen on `SelectionController` only — the controller is the single source of truth for `current_step`, `selected_residue`, and `comparison_steps`. Adding a new linked view is a matter of subscribing to the relevant signals.

## Data formats

<!-- AUTO-GENERATED: derived from data.py load_run() and actual log TSV headers -->
### Metrics TSV (`--tsv`)

Tab-separated, one row per step. Columns used (others are ignored):

| Column | Type | Notes |
|--------|------|-------|
| `step` | int | 0-based TTT step index |
| `loss` | float | Training loss; empty/NaN at step 0 |
| `plddt` | float | Mean pLDDT across all atoms (matches `calculate_plddt`) |
| `lddt` | float | lDDT score |

Example header:
```
step  accumulated_step  loss  perplexity  ttt_step_time  score_seq_time  eval_step_time  plddt  tm_score  lddt
```

### PDB folder (`--pdbs`)

Files must be named `step_<i>.pdb` (any integer suffix). Per-residue pLDDT is read from the **B-factor column (cols 60–66, 0-indexed)** of every `ATOM` record where the atom name (cols 12–15) is `CA`. The number of residues must be constant across all steps.

### `PtttRun` fields (output of `load_run`)

| Field | Shape / type | Description |
|-------|-------------|-------------|
| `steps` | `(S,) int32` | Step indices from TSV |
| `loss` | `(S,) float64` | NaN where missing |
| `plddt_mean` | `(S,) float64` | From TSV `plddt` column |
| `lddt` | `(S,) float64` | NaN where missing |
| `plddt_matrix` | `(S, N) float32` | Per-residue pLDDT from CA B-factors |
| `plddt_delta` | `(S, N) float32` | `plddt_matrix − plddt_matrix[0]` |
| `n_steps` | `int` | Number of steps S |
| `n_residues` | `int` | Number of residues N |
| `best_step` | `int` | `argmax(plddt_mean)` |
| `best_plddt` | `float` | `plddt_mean[best_step]` |
<!-- END AUTO-GENERATED -->

### Bundled data

`data/summary.csv` lists the proteins shipped with the repo:

| ID | Length | Base pLDDT | Version |
|----|--------|------------|---------|
| A0A6J5N0Y1 | 62 | 91.47 | BASE |
| A5A3S1 | 91 | 90.81 | BASE+LOGAN+12CY |
| A0A646QXE5 | 89 | — | — |

Each has a corresponding `data/logs/<ID>_log.tsv` and `data/logs/<ID>_pdbs/` folder.

## Technical notes

- **Heatmap performance.** A single vectorized NumPy pass maps the *(S × N)* pLDDT matrix to RGBA via lookup tables in `colors.py`, then writes the result into a `QImage` buffer. The image is wrapped in a `QGraphicsPixmapItem` — Qt handles all subsequent scaling and panning on the GPU.
- **Brushing & linking.** Strict pub/sub through `SelectionController`. No view holds a reference to any other view.
- **Minimal PDB parsing.** Reads CA B-factors from fixed columns (60–66) of `ATOM` records. ~30 lines, no external dependency.
- **OpenGL scatter.** The embedding view drops to `QOpenGLWidget` with a custom `QGraphicsItem` for the points layer, so panning and zooming through 100s of residues stays at 60 FPS.
- **Axis ticks.** A `nice_ticks()` helper picks human-friendly tick steps (1, 2, 5 × 10ⁿ) instead of using a charting library's defaults.

## Background

**pLDDT** — predicted Local Distance Difference Test, the per-residue confidence score output by AlphaFold-style structure predictors (0–100; >90 = very high confidence).
**lDDT** — ground-truth per-residue structural similarity to a reference structure, when available.
**Test-time training (TTT)** — fine-tuning the protein language model on a single target sequence at inference time, in the hope that this raises confidence and accuracy for that specific protein.

The visualization makes it possible to see *which residues* drive an overall pLDDT improvement — often only a few flexible loops change while the structured core stays fixed — turning a single scalar trajectory into a residue-resolved story.
