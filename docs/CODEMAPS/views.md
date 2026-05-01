<!-- Generated: 2026-05-01 | Files scanned: 7 | LOC: ~2453 -->

# Views (T1–T6)

All views consume `PtttRun` (read-only) + `SelectionController` (signals). Each implements one or more of: `set_run`, `set_step`, `set_residue_range`. Views never mutate `PtttRun`.

## Layout

```
┌──────────────── MainWindow ─────────────────┬──────────────┐
│ ┌──────────── (T1) ───────┬── (T4) ──────┐  │              │
│ │ LineChartView           │ EmbeddingView│  │   T6         │
│ ├──────────── (T2) ───────┼── (T3) ──────┤  │ Residue      │
│ │ HeatmapView             │ ProfileView  │  │ DetailDock   │
│ └─────────────────────────┴──────────────┘  │ (right side, │
│                                             │  hidden if   │
│                                             │  selected=-1)│
└─────────────────────────────────────────────┴──────────────┘
```

## T1 — LineChartView ([views/line_chart.py](views/line_chart.py))

Multi-panel line chart of global metrics across S steps.
- Panels: loss, plddt_mean, lddt; SS-stratified mean overlays toggleable.
- Click a panel → `setCurrentStep`; current-step indicator follows.
- `_ss_stratified_plddt_means(run)` → (3, S) means by class for dashed overlays.

## T2 — HeatmapView ([views/heatmap.py](views/heatmap.py))

(S × N) pLDDT heatmap rendered as a single ARGB32 `QImage` (no per-cell items).
- Hover → `setHoveredResidue`; click → `setCurrentStep` + `setSelectedResidue`.
- Topped by a `SecondaryStructureTrack` aligned to the residue X axis.
- SS-class filter applied via translucent white overlay rects over out-of-class segments.

## T3 — ProfileView ([views/profile_view.py](views/profile_view.py))

Overlaid per-residue pLDDT profiles + step picker.
- Comparison-step selector layers multiple profile lines with categorical palette.
- Same SS-track-on-top + overlay-rect fade pattern as T2.

## T4 — EmbeddingView ([views/embedding_view.py](views/embedding_view.py))

2D scatter of per-residue embeddings (PCA/UMAP/t-SNE joint fit) animated by step.
- One `PointsItem` for current step; recolored by `set_color_mode("plddt"|"ss")`.
- Hover → brute-force argmin in scene space; click → select.
- SS filter via `PointsItem.set_alpha_mask(np.isin(ss_row, allowed))`.
- Reduction is computed once via `reduction.reduce_joint`, cached on disk.

## T5 — Range brushing & linking

Specced as out-of-class fade across views. Currently implemented only as the SS-class checkbox filter (Phase F). Range-brush rectangle selection is **not wired**.

## T6 — ResidueDetailDock ([views/residue_detail.py](views/residue_detail.py))

Right-docked widget with four stacked sub-panels for the selected residue:

| Sub-panel | Class | Source |
|---|---|---|
| pLDDT trajectory across S | `_PlddtTrajectoryView` | `plddt_matrix[:, r]` |
| SS evolution strip | `_SsEvolutionView` | `ss_matrix[:, r]` |
| Embedding trajectory polyline | `_EmbeddingTrajectoryView` | `coords_2d[:, r, :]` (provider callback) |
| ±5 sequence-context cells | `_SequenceContextView` | `aa_sequence` + `plddt_matrix[step]` |

Click a step on either trajectory → `setCurrentStep`. Click a neighbor cell → `setSelectedResidue`. Constructor takes `coords_2d_provider: Callable[[], np.ndarray | None]` so the dock pulls fresh reduced coords from `EmbeddingView` rather than caching its own copy.

## Shared widgets

- [views/ss_track.py](views/ss_track.py) `SecondaryStructureTrack` — colored segments via `data.ss_segments`, hover tooltip with class/length, drives `setHoveredResidue` for X-axis alignment.
- [points_item.py](points_item.py) `PointsItem` — pre-groups points by ARGB color, draws ellipses per group; supports `set_alpha_mask`.
- [chart_axes.py](chart_axes.py) `draw_axes`, `nice_ticks` — used by every chart view.
