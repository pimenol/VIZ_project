"""T4 - 2D scatter view of per-residue embeddings, animated across steps."""

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QSurfaceFormat,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chart_axes import draw_axes, nice_ticks
from colors import (
    SS_COLORS,
    SS_LETTERS,
    alphafold_color_array,
    ss_color_array,
)
from controller import SelectionController
from data import PtttRun
from points_item import PointsItem
from reduction import ReductionResult, reduce_joint

_LEFT = 52.0
_TOP = 28.0
_RIGHT_MARGIN = 16.0
_BOT_MARGIN = 28.0
_PLOT_W = 520.0
_PLOT_H = 380.0
_TOTAL_W = _LEFT + _PLOT_W + _RIGHT_MARGIN
_TOTAL_H = _TOP + _PLOT_H + _BOT_MARGIN
_PLOT_RECT = QRectF(_LEFT, _TOP, _PLOT_W, _PLOT_H)

_HOVER_PIX_RADIUS = 8.0
_POINT_RADIUS = 4.0


class EmbeddingScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self.setSceneRect(0, 0, _TOTAL_W, _TOTAL_H)
        self.setBackgroundBrush(QColor(252, 252, 252))

        self._run = run
        self._reduction: ReductionResult | None = None
        self._coords_scene: np.ndarray = np.empty((0, 0, 2), dtype=np.float32)
        self._x_lo = self._x_hi = 0.0
        self._y_lo = self._y_hi = 0.0
        self._method = "pca"
        self._current_step = 0
        self._color_mode = "plddt"
        self._legend_items: list = []
        self._ss_filter: set[int] = {0, 1, 2}
        self._res_lo = 0
        self._res_hi = run.n_residues - 1

        self._points = PointsItem(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            radius=_POINT_RADIUS,
        )
        self._points.setZValue(5)
        self.addItem(self._points)

        self._axes = None
        self._title = self.addText("")
        self._title.setDefaultTextColor(QColor(60, 60, 60))
        self._var_label = self.addText("")
        self._var_label.setDefaultTextColor(QColor(120, 120, 120))

        self.set_run(run)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._current_step = 0
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._compute_reduction()
        self._refresh_step()

    def set_current_step(self, step: int) -> None:
        if 0 <= step < self._run.n_steps:
            self._current_step = step
            self._refresh_step()

    def set_color_mode(self, mode: str) -> None:
        if mode == self._color_mode:
            return
        self._color_mode = mode
        self._refresh_step()
        self._rebuild_legend()

    def set_ss_filter(self, allowed: set[int]) -> None:
        self._ss_filter = set(allowed)
        self._apply_ss_filter()

    def set_residue_range(self, lo: int, hi: int) -> None:
        if (lo, hi) == (self._res_lo, self._res_hi):
            return
        self._res_lo = lo
        self._res_hi = hi
        self._apply_ss_filter()

    def coords_2d_data(self) -> np.ndarray | None:
        return self._reduction.coords_2d if self._reduction is not None else None

    def _compute_reduction(self) -> None:
        cache_dir: Path | None = None
        cache_key = f"{self._run.embedding_kind}_{self._run.n_steps}_{self._run.n_residues}_{self._run.embeddings_hd.shape[2]}"
        self._reduction = reduce_joint(
            self._run.embeddings_hd,
            method=self._method,
            cache_dir=cache_dir,
            cache_key=cache_key,
        )
        coords_2d = self._reduction.coords_2d
        # Compute global ranges (over all steps) so axes are stable as the slider moves.
        self._x_lo = float(coords_2d[..., 0].min())
        self._x_hi = float(coords_2d[..., 0].max())
        self._y_lo = float(coords_2d[..., 1].min())
        self._y_hi = float(coords_2d[..., 1].max())
        # Pad ranges by 5% so points aren't on the border.
        x_pad = (self._x_hi - self._x_lo) * 0.05 or 1.0
        y_pad = (self._y_hi - self._y_lo) * 0.05 or 1.0
        self._x_lo -= x_pad
        self._x_hi += x_pad
        self._y_lo -= y_pad
        self._y_hi += y_pad

        self._coords_scene = self._to_scene_coords(coords_2d)
        self._rebuild_axes()
        self._update_var_label()

    def _to_scene_coords(self, coords_2d: np.ndarray) -> np.ndarray:
        # Y inverted: data y_lo at bottom of rect, y_hi at top.
        x = coords_2d[..., 0]
        y = coords_2d[..., 1]
        sx = _LEFT + (x - self._x_lo) / (self._x_hi - self._x_lo) * _PLOT_W
        sy = _TOP + _PLOT_H - (y - self._y_lo) / (self._y_hi - self._y_lo) * _PLOT_H
        return np.stack([sx, sy], axis=-1).astype(np.float32)

    def _rebuild_axes(self) -> None:
        if self._axes is not None:
            for items in (
                self._axes.x_ticks, self._axes.y_ticks,
                self._axes.x_labels, self._axes.y_labels,
                self._axes.x_gridlines, self._axes.y_gridlines,
                self._axes.border,
            ):
                for it in items:
                    self.removeItem(it)
        x_ticks = nice_ticks(self._x_lo, self._x_hi, target=6)
        y_ticks = nice_ticks(self._y_lo, self._y_hi, target=5)
        self._axes = draw_axes(
            self, _PLOT_RECT, x_ticks, y_ticks,
            self._x_lo, self._x_hi, self._y_lo, self._y_hi,
        )

    def _update_var_label(self) -> None:
        if self._method == "pca" and self._reduction and self._reduction.explained_variance_ratio:
            a, b = self._reduction.explained_variance_ratio
            text = f"PC1: {a*100:.1f}%   PC2: {b*100:.1f}%"
        elif self._method == "umap":
            text = "UMAP-1 / UMAP-2"
        elif self._method == "tsne":
            text = "t-SNE 1 / t-SNE 2"
        else:
            text = ""
        self._var_label.setPlainText(text)
        self._var_label.setPos(_LEFT + 4, _TOP + _PLOT_H + 6)
        self._var_label.setZValue(11)

    def _refresh_step(self) -> None:
        if self._coords_scene.size == 0:
            return
        s = self._current_step
        coords = self._coords_scene[s]                       # (N, 2)
        if self._color_mode == "ss":
            colors = ss_color_array(self._run.ss_matrix[s])
        else:
            colors = alphafold_color_array(self._run.plddt_matrix[s])
        self._points.set_data(coords, colors)
        self._title.setPlainText(f"Embedding 2D — step {s} ({self._method.upper()})")
        self._title.setPos(_LEFT, 6)
        self._title.setZValue(11)
        self._apply_ss_filter()

    def _apply_ss_filter(self) -> None:
        if self._coords_scene.size == 0:
            return
        ss_row = self._run.ss_matrix[self._current_step]
        ss_mask = np.isin(ss_row, list(self._ss_filter))
        res_idx = np.arange(self._run.n_residues)
        range_mask = (res_idx >= self._res_lo) & (res_idx <= self._res_hi)
        combined = ss_mask & range_mask
        if combined.all():
            self._points.set_alpha_mask(None)
        else:
            self._points.set_alpha_mask(combined)

    def _rebuild_legend(self) -> None:
        for item in self._legend_items:
            self.removeItem(item)
        self._legend_items = []

        if self._color_mode != "ss":
            return

        font = QFont()
        font.setPointSize(7)
        sw = 10.0
        gap = 4.0
        x = _LEFT + _PLOT_W - 100.0
        y = _TOP + 4.0
        for label in (0, 1, 2):
            rect = QGraphicsRectItem(x, y, sw, sw)
            rect.setBrush(QBrush(SS_COLORS[label]))
            rect.setPen(QPen(QColor(80, 80, 80), 0.5))
            rect.setZValue(11)
            self.addItem(rect)
            self._legend_items.append(rect)
            txt = QGraphicsSimpleTextItem(SS_LETTERS[label])
            txt.setFont(font)
            txt.setBrush(QColor(50, 50, 50))
            txt.setPos(x + sw + 2.0, y - 2.0)
            txt.setZValue(11)
            self.addItem(txt)
            self._legend_items.append(txt)
            x += sw + 2.0 + 14.0 + gap

    def residue_at(self, scene_point: QPointF) -> int:
        return self._points.index_at(scene_point, _HOVER_PIX_RADIUS)


class EmbeddingView(QWidget):
    def __init__(self, run: PtttRun, ctrl: SelectionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._run = run

        self._scene = EmbeddingScene(run)

        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        gl = QOpenGLWidget()
        gl.setFormat(fmt)

        self._gview = QGraphicsView(self._scene)
        self._gview.setViewport(gl)
        self._gview.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
        )
        self._gview.setBackgroundBrush(QColor(240, 240, 240))
        self._gview.setDragMode(QGraphicsView.ScrollHandDrag)
        self._gview.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._gview.setMouseTracking(True)
        self._gview.mouseMoveEvent = self._view_mouse_move
        self._gview.mousePressEvent = self._view_mouse_press
        self._gview.wheelEvent = self._view_wheel

        toolbar = QWidget(self)
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(4, 2, 4, 2)
        tb.setSpacing(4)
        tb.addWidget(QLabel("Color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["pLDDT", "Secondary structure"])
        self._color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        tb.addWidget(self._color_combo)
        tb.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self._gview)

        ctrl.currentStepChanged.connect(self._scene.set_current_step)
        ctrl.ssClassFilterChanged.connect(self._scene.set_ss_filter)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._scene.set_run(run)

    def set_residue_range(self, lo: int, hi: int) -> None:
        self._scene.set_residue_range(lo, hi)

    def _on_color_mode_changed(self, text: str) -> None:
        mode = "ss" if text == "Secondary structure" else "plddt"
        self._scene.set_color_mode(mode)

    def coords_2d_data(self) -> np.ndarray | None:
        return self._scene.coords_2d_data()

    def _view_mouse_move(self, event) -> None:
        sp = self._gview.mapToScene(event.pos())
        res = self._scene.residue_at(sp)
        if res >= 0:
            self._ctrl.setHoveredResidue(res)
            plddt = float(self._run.plddt_matrix[self._ctrl.current_step, res])
            QToolTip.showText(event.globalPos(), f"Res {res}\npLDDT {plddt:.1f}")
        else:
            self._ctrl.setHoveredResidue(-1)
            QToolTip.hideText()
        QGraphicsView.mouseMoveEvent(self._gview, event)

    def _view_mouse_press(self, event) -> None:
        sp = self._gview.mapToScene(event.pos())
        res = self._scene.residue_at(sp)
        if res >= 0:
            self._ctrl.setSelectedResidue(res)
        QGraphicsView.mousePressEvent(self._gview, event)

    def _view_wheel(self, event) -> None:
        factor = 1 + event.angleDelta().y() * 0.001
        self._gview.scale(factor, factor)
