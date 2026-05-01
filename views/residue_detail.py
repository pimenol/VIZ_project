"""T6 — per-residue detail dock with header, pLDDT trajectory, embedding trajectory, sequence context."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDockWidget,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chart_axes import draw_axes, nice_ticks
from colors import alphafold_color_array
from controller import SelectionController
from data import PtttRun

# Plot dims for the two mini-charts (pLDDT trajectory, embedding trajectory).
_MINI_W = 320.0
_MINI_H = 110.0
_MINI_LEFT = 38.0
_MINI_TOP = 14.0
_MINI_RIGHT = 10.0
_MINI_BOT = 22.0
_MINI_PLOT_W = _MINI_W - _MINI_LEFT - _MINI_RIGHT
_MINI_PLOT_H = _MINI_H - _MINI_TOP - _MINI_BOT
_MINI_PLOT = QRectF(_MINI_LEFT, _MINI_TOP, _MINI_PLOT_W, _MINI_PLOT_H)

# Sequence context strip dims (11 cells horizontally).
_CTX_RADIUS = 5
_CTX_CELLS = 2 * _CTX_RADIUS + 1
_CTX_CELL_W = 26.0
_CTX_CELL_H = 30.0
_CTX_W = _CTX_CELLS * _CTX_CELL_W + 4
_CTX_H = _CTX_CELL_H + 22

_REF_LEVELS = (50.0, 70.0, 90.0)
_INDICATOR_COLOR = QColor(255, 80, 0)
_TRAJECTORY_COLOR = QColor(60, 80, 200)


class _PlddtTrajectoryView(QGraphicsView):
    """pLDDT vs step for one residue."""

    def __init__(self, run: PtttRun, ctrl: SelectionController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run = run
        self._ctrl = ctrl
        self._residue = -1
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, _MINI_W, _MINI_H)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor(252, 252, 252))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(int(_MINI_H + 4))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._path_item: QGraphicsPathItem | None = None
        self._indicator: QGraphicsLineItem | None = None
        self._axes = None

        ctrl.currentStepChanged.connect(self._on_current_step)

    def set_run(self, run: PtttRun) -> None:
        self._run = run

    def set_residue(self, residue: int) -> None:
        self._residue = residue
        self._rebuild()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        sp = self.mapToScene(event.pos())
        if _MINI_PLOT.contains(sp):
            n = self._run.n_steps
            x_lo = 0.0
            x_hi = max(n - 1, 1)
            t = (sp.x() - _MINI_LEFT) / _MINI_PLOT_W
            step = int(round(x_lo + t * (x_hi - x_lo)))
            step = max(0, min(n - 1, step))
            self._ctrl.setCurrentStep(step)
        super().mousePressEvent(event)

    def _rebuild(self) -> None:
        self._scene.clear()
        self._path_item = None
        self._indicator = None
        self._axes = None
        if self._residue < 0 or self._residue >= self._run.n_residues:
            return

        n = self._run.n_steps
        y = self._run.plddt_matrix[:, self._residue]
        x_lo, x_hi = 0.0, max(n - 1, 1)
        y_lo, y_hi = 0.0, 100.0
        x_ticks = nice_ticks(x_lo, x_hi, target=4)
        y_ticks = (0, 50, 100)
        self._axes = draw_axes(
            self._scene, _MINI_PLOT, x_ticks, list(y_ticks),
            x_lo, x_hi, y_lo, y_hi,
            font_size=7,
        )

        # Reference dashed lines at 50/70/90.
        ref_pen = QPen(QColor(180, 180, 180), 0.8, Qt.DashLine)
        ref_pen.setCosmetic(True)
        for level in _REF_LEVELS:
            sy = self._y_to_scene(level)
            ln = QGraphicsLineItem(_MINI_LEFT, sy, _MINI_LEFT + _MINI_PLOT_W, sy)
            ln.setPen(ref_pen)
            ln.setZValue(8)
            self._scene.addItem(ln)

        # Trajectory polyline.
        path = QPainterPath()
        started = False
        for s in range(n):
            v = float(y[s])
            if not np.isfinite(v):
                started = False
                continue
            sx = self._x_to_scene(float(s))
            sy = self._y_to_scene(v)
            if not started:
                path.moveTo(sx, sy)
                started = True
            else:
                path.lineTo(sx, sy)
        item = QGraphicsPathItem(path)
        pen = QPen(_TRAJECTORY_COLOR, 1.6)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(10)
        self._scene.addItem(item)
        self._path_item = item

        # Current-step indicator.
        ind_pen = QPen(_INDICATOR_COLOR, 1.2)
        ind_pen.setCosmetic(True)
        self._indicator = QGraphicsLineItem(0, _MINI_TOP, 0, _MINI_TOP + _MINI_PLOT_H)
        self._indicator.setPen(ind_pen)
        self._indicator.setZValue(12)
        self._scene.addItem(self._indicator)
        self._move_indicator(self._ctrl.current_step)

    def _x_to_scene(self, step: float) -> float:
        n_max = max(self._run.n_steps - 1, 1)
        return _MINI_LEFT + (step / n_max) * _MINI_PLOT_W

    def _y_to_scene(self, v: float) -> float:
        return _MINI_TOP + _MINI_PLOT_H - (v / 100.0) * _MINI_PLOT_H

    def _move_indicator(self, step: int) -> None:
        if self._indicator is None:
            return
        sx = self._x_to_scene(float(step))
        self._indicator.setLine(sx, _MINI_TOP, sx, _MINI_TOP + _MINI_PLOT_H)

    def _on_current_step(self, step: int) -> None:
        self._move_indicator(step)


class _EmbeddingTrajectoryView(QGraphicsView):
    """Polyline through embeddings_2d[:, residue, :] across all steps."""

    def __init__(
        self,
        run: PtttRun,
        coords_2d_provider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run = run
        self._coords_2d_provider = coords_2d_provider
        self._residue = -1
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, _MINI_W, _MINI_H)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor(252, 252, 252))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(int(_MINI_H + 4))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_run(self, run: PtttRun) -> None:
        self._run = run

    def set_residue(self, residue: int) -> None:
        self._residue = residue
        self._rebuild()

    def _rebuild(self) -> None:
        self._scene.clear()
        if self._residue < 0:
            return
        coords = self._coords_2d_provider()
        if coords is None or coords.size == 0:
            return
        traj = coords[:, self._residue, :]                  # (S, 2)
        x_lo = float(traj[:, 0].min())
        x_hi = float(traj[:, 0].max())
        y_lo = float(traj[:, 1].min())
        y_hi = float(traj[:, 1].max())
        x_pad = (x_hi - x_lo) * 0.10 or 1.0
        y_pad = (y_hi - y_lo) * 0.10 or 1.0
        x_lo -= x_pad
        x_hi += x_pad
        y_lo -= y_pad
        y_hi += y_pad

        x_ticks = nice_ticks(x_lo, x_hi, target=3)
        y_ticks = nice_ticks(y_lo, y_hi, target=3)
        draw_axes(
            self._scene, _MINI_PLOT, x_ticks, y_ticks,
            x_lo, x_hi, y_lo, y_hi, font_size=7,
        )

        def to_scene(p: np.ndarray) -> tuple[float, float]:
            sx = _MINI_LEFT + (p[0] - x_lo) / (x_hi - x_lo) * _MINI_PLOT_W
            sy = _MINI_TOP + _MINI_PLOT_H - (p[1] - y_lo) / (y_hi - y_lo) * _MINI_PLOT_H
            return sx, sy

        # Polyline.
        path = QPainterPath()
        sx0, sy0 = to_scene(traj[0])
        path.moveTo(sx0, sy0)
        for i in range(1, len(traj)):
            sx, sy = to_scene(traj[i])
            path.lineTo(sx, sy)
        line_item = QGraphicsPathItem(path)
        pen = QPen(_TRAJECTORY_COLOR, 1.6)
        pen.setCosmetic(True)
        line_item.setPen(pen)
        line_item.setZValue(10)
        self._scene.addItem(line_item)

        # Arrowhead at last segment.
        if len(traj) >= 2:
            self._scene.addItem(self._make_arrowhead(to_scene(traj[-2]), to_scene(traj[-1])))

        # Step labels at 0, S//2, S-1.
        s_max = len(traj) - 1
        for s in {0, s_max // 2, s_max}:
            sx, sy = to_scene(traj[s])
            dot = QGraphicsRectItem(sx - 2.0, sy - 2.0, 4.0, 4.0)
            dot.setBrush(QBrush(_TRAJECTORY_COLOR))
            dot.setPen(QPen(Qt.NoPen))
            dot.setZValue(11)
            self._scene.addItem(dot)
            label = QGraphicsSimpleTextItem(str(s))
            f = QFont()
            f.setPointSize(7)
            label.setFont(f)
            label.setBrush(QColor(60, 60, 60))
            label.setPos(sx + 4.0, sy - 12.0)
            label.setZValue(12)
            self._scene.addItem(label)

    @staticmethod
    def _make_arrowhead(p0: tuple[float, float], p1: tuple[float, float]) -> QGraphicsPolygonItem:
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length
        size = 6.0
        # Two flank points at ±60° behind tip.
        px, py = -uy, ux
        bx = x1 - ux * size
        by = y1 - uy * size
        a = QPointF(x1, y1)
        b = QPointF(bx + px * size * 0.5, by + py * size * 0.5)
        c = QPointF(bx - px * size * 0.5, by - py * size * 0.5)
        poly = QGraphicsPolygonItem(QPolygonF([a, b, c]))
        poly.setBrush(QBrush(_TRAJECTORY_COLOR))
        poly.setPen(QPen(Qt.NoPen))
        poly.setZValue(11)
        return poly


class _SequenceContextView(QGraphicsView):
    """11-cell strip showing residue [i-5..i+5] colored by current step's pLDDT."""

    def __init__(self, run: PtttRun, ctrl: SelectionController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run = run
        self._ctrl = ctrl
        self._residue = -1
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, _CTX_W, _CTX_H)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor(252, 252, 252))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(int(_CTX_H + 4))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        ctrl.currentStepChanged.connect(self._on_current_step)

    def set_run(self, run: PtttRun) -> None:
        self._run = run

    def set_residue(self, residue: int) -> None:
        self._residue = residue
        self._rebuild()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        sp = self.mapToScene(event.pos())
        if 0 <= sp.y() <= _CTX_CELL_H + 2 and self._residue >= 0:
            offset = int(sp.x() // _CTX_CELL_W) - _CTX_RADIUS
            target = self._residue + offset
            if 0 <= target < self._run.n_residues:
                self._ctrl.setSelectedResidue(target)
        super().mousePressEvent(event)

    def _rebuild(self) -> None:
        self._scene.clear()
        if self._residue < 0:
            return
        i = self._residue
        s = self._ctrl.current_step
        plddt_row = self._run.plddt_matrix[s]
        seq = self._run.aa_sequence
        for k in range(_CTX_CELLS):
            j = i - _CTX_RADIUS + k
            x = k * _CTX_CELL_W + 2.0
            if not (0 <= j < self._run.n_residues):
                continue
            col_argb = int(alphafold_color_array(np.array([plddt_row[j]], dtype=np.float32))[0])
            cell = QGraphicsRectItem(x, 4.0, _CTX_CELL_W - 2.0, _CTX_CELL_H)
            cell.setBrush(QBrush(QColor.fromRgba(col_argb)))
            border = QPen(QColor(120, 120, 120) if j != i else QColor(255, 80, 0), 1.0 if j != i else 2.0)
            border.setCosmetic(True)
            cell.setPen(border)
            cell.setZValue(5)
            self._scene.addItem(cell)
            letter = seq[j] if 0 <= j < len(seq) else "X"
            txt = QGraphicsSimpleTextItem(letter)
            f = QFont()
            f.setPointSize(11)
            f.setBold(True)
            txt.setFont(f)
            br = txt.boundingRect()
            txt.setPos(x + (_CTX_CELL_W - 2.0 - br.width()) * 0.5, 4.0 + (_CTX_CELL_H - br.height()) * 0.5)
            txt.setBrush(QColor(20, 20, 20))
            txt.setZValue(6)
            self._scene.addItem(txt)
            # Index label below.
            idx = QGraphicsSimpleTextItem(str(j))
            f2 = QFont()
            f2.setPointSize(7)
            idx.setFont(f2)
            ibr = idx.boundingRect()
            idx.setPos(x + (_CTX_CELL_W - 2.0 - ibr.width()) * 0.5, 4.0 + _CTX_CELL_H + 1.0)
            idx.setBrush(QColor(120, 120, 120))
            self._scene.addItem(idx)

    def _on_current_step(self, _step: int) -> None:
        self._rebuild()


class ResidueDetailDock(QDockWidget):
    """Top-level dock that swaps content on residueSelectedChanged."""

    def __init__(
        self,
        run: PtttRun,
        ctrl: SelectionController,
        coords_2d_provider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Residue detail", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self._run = run
        self._ctrl = ctrl
        self._residue = -1

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = QLabel("(no residue selected)")
        f = QFont()
        f.setBold(True)
        self._header.setFont(f)
        layout.addWidget(self._header)

        self._stats = QLabel("")
        self._stats.setStyleSheet("color: #555;")
        layout.addWidget(self._stats)

        layout.addWidget(QLabel("pLDDT trajectory"))
        self._plddt_chart = _PlddtTrajectoryView(run, ctrl)
        layout.addWidget(self._plddt_chart)

        layout.addWidget(QLabel("Embedding trajectory"))
        self._emb_chart = _EmbeddingTrajectoryView(run, coords_2d_provider)
        layout.addWidget(self._emb_chart)

        layout.addWidget(QLabel("Sequence context (±5)"))
        self._seq_strip = _SequenceContextView(run, ctrl)
        layout.addWidget(self._seq_strip)

        layout.addStretch(1)
        self.setWidget(inner)
        self.setMinimumWidth(360)

        ctrl.residueSelectedChanged.connect(self.set_residue)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._plddt_chart.set_run(run)
        self._emb_chart.set_run(run)
        self._seq_strip.set_run(run)
        self.set_residue(-1)

    def set_residue(self, residue: int) -> None:
        self._residue = residue
        if residue < 0:
            self.hide()
            return
        self._update_header()
        self._plddt_chart.set_residue(residue)
        self._emb_chart.set_residue(residue)
        self._seq_strip.set_residue(residue)
        self.show()

    def _update_header(self) -> None:
        i = self._residue
        seq = self._run.aa_sequence
        letter = seq[i] if 0 <= i < len(seq) else "X"
        self._header.setText(f"Residue {i}  ({letter})")

        col = self._run.plddt_matrix[:, i]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            self._stats.setText("no finite pLDDT")
            return
        v_min = float(finite.min())
        v_max = float(finite.max())
        v_mean = float(finite.mean())
        delta = float(col[-1] - col[0]) if np.isfinite(col[0]) and np.isfinite(col[-1]) else float("nan")
        self._stats.setText(
            f"pLDDT  min {v_min:.1f}  ·  mean {v_mean:.1f}  ·  max {v_max:.1f}  ·  Δ(end−start) {delta:+.1f}"
        )
