# ProteinTTT Visualization

Desktop tool for exploring how a protein-language-model's structure prediction evolves across test-time-training (TTT) optimization steps.

## Requirements

- Python 3.10+
- PySide6
- numpy

```bash
pip install PySide6 numpy
```

## Usage

```bash
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

| View | What it shows |
|------|---------------|
| **T1 Line chart** (top) | Mean pLDDT, perplexity, TM-score, lDDT across steps. Peak pLDDT step marked in red. |
| **T2 Heatmap** (bottom-left) | Per-residue pLDDT across all steps. Color modes: AlphaFold bands or delta-from-step-0. |
| **T3 Profile** (bottom-right) | Overlaid per-residue pLDDT polylines for selected comparison steps. Step picker on the right. |

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

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| ←/→ | Previous/next step |
| Shift+←/→ | Jump 10 steps |
| Home | Step 0 |
| End | Best pLDDT step |
| +/- | Zoom in/out (heatmap) |
| R | Reset zoom |
| F | Fit to view |

## Toolbar

- **Load…** — pick a metrics TSV then a PDB folder
- **Demo** — reload synthetic data
- **Step slider/spinbox** — navigate steps
- **Color** — toggle heatmap between AlphaFold bands and delta-from-step-0
- **Res:** — filter displayed residue range
- **Save PNG…** — export current heatmap scene to PNG

## File structure

```
main.py           MainWindow, toolbar, CLI
data.py           PtttRun dataclass + load_run()
controller.py     SelectionController (shared signals)
colors.py         Vectorized AlphaFold / delta palettes
chart_axes.py     nice_ticks() + draw_axes() helpers
synthetic.py      make_demo_run() for development
views/
  heatmap.py      T2 — QImage-backed heatmap
  line_chart.py   T1 — multi-panel line chart
  profile_view.py T3 — overlaid profiles + step picker
```

## Data formats

<!-- AUTO-GENERATED: derived from data.py load_run() and actual log TSV headers -->
### Metrics TSV (`--tsv`)

Tab-separated, one row per step. Columns used (others are ignored):

| Column | Type | Notes |
|--------|------|-------|
| `step` | int | 0-based TTT step index |
| `loss` | float | Training loss; empty/NaN at step 0 |
| `perplexity` | float | Sequence perplexity; empty/NaN at step 0 |
| `plddt` | float | Mean pLDDT across all atoms (matches `calculate_plddt`) |
| `tm_score` | float | TM-score vs reference; may be all-empty for a run |
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
| `perplexity` | `(S,) float64` | NaN where missing |
| `plddt_mean` | `(S,) float64` | From TSV `plddt` column |
| `tm_score` | `(S,) float64` | NaN where missing |
| `lddt` | `(S,) float64` | NaN where missing |
| `plddt_matrix` | `(S, N) float32` | Per-residue pLDDT from CA B-factors |
| `plddt_delta` | `(S, N) float32` | `plddt_matrix − plddt_matrix[0]` |
| `n_steps` | `int` | Number of steps S |
| `n_residues` | `int` | Number of residues N |
| `best_step` | `int` | `argmax(plddt_mean)` |
| `best_plddt` | `float` | `plddt_mean[best_step]` |
<!-- END AUTO-GENERATED -->

### Included data

`data/summary.csv` lists the 3 proteins shipped with the project:

| ID | Length | Base pLDDT | Version |
|----|--------|------------|---------|
| A0A6J5N0Y1 | 62 | 91.47 | BASE |
| A5A3S1 | 91 | 90.81 | BASE+LOGAN+12CY |
| A0A646QXE5 | 89 | — | — |

Each has a corresponding `data/logs/<ID>_log.tsv` and `data/logs/<ID>_pdbs/` folder.

## Implementation notes

- **No charting libraries.** Every graphical element (curves, axes, heatmap pixels) is built from PySide6 primitives: `QPainterPath`, `QGraphicsPathItem`, `QGraphicsLineItem`, `QImage`.
- **Heatmap performance.** The entire (steps × residues) grid is rendered into a single `QImage` in one vectorized numpy pass and wrapped in a `QGraphicsPixmapItem`. Color-mode toggle refills the image buffer without rebuilding the scene.
- **Brushing & linking.** All views communicate exclusively through `SelectionController` signals — never directly to each other.
- **PDB parsing** is a minimal hand-written parser: reads CA-atom B-factors from fixed PDB column positions (60:66).
