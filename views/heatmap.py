"""T2 — per-residue pLDDT heatmap backed by a single QImage."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QPen,
    QPixmap,
    QSurfaceFormat,
    QPainter,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QToolTip,
)

from chart_axes import nice_ticks
from colors import alphafold_color_array, delta_color_array
from controller import SelectionController
from data import PtttRun

# Layout constants (scene coords)
_LEFT_MARGIN = 52.0
_RIGHT_MARGIN = 10.0
_TOP_MARGIN = 10.0
_BOT_MARGIN = 28.0
_LEGEND_W = 18.0
_LEGEND_H = 120.0
_LEGEND_GAP = 8.0

_STEP_OVL_COLOR = QColor(255, 80, 0)     # current-step horizontal overlay
_RES_OVL_COLOR = QColor(0, 100, 220)     # selected-residue vertical overlay


class HeatmapScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self._run = run
        self._mode: Literal["absolute", "delta"] = "absolute"
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._step_line: QGraphicsLineItem | None = None
        self._res_line: QGraphicsLineItem | None = None
        self._axis_items: list[QGraphicsItem] = []
        self._legend_items: list[QGraphicsItem] = []
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._pixmap_item = None
        self._step_line = None
        self._res_line = None
        self._axis_items = []
        self._legend_items = []
        self.clear()
        self._build()

    def set_color_mode(self, mode: Literal["absolute", "delta"]) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._refill_image()
        self._rebuild_legend()

    def set_residue_range(self, lo: int, hi: int) -> None:
        self._res_lo = lo
        self._res_hi = hi
        self._refill_image()
        self._rebuild_axes()

    def move_step_line(self, step: int) -> None:
        if self._step_line is None:
            return
        y = self._step_to_scene_y(step)
        pr = self._plot_rect()
        self._step_line.setLine(pr.left(), y, pr.right(), y)

    def move_res_line(self, residue: int) -> None:
        if self._res_line is None:
            return
        if residue < 0:
            self._res_line.setVisible(False)
            return
        self._res_line.setVisible(True)
        x = self._res_to_scene_x(residue)
        pr = self._plot_rect()
        self._res_line.setLine(x, pr.top(), x, pr.bottom())

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def _build(self) -> None:
        run = self._run
        pr = self._plot_rect()

        # --- QImage pixmap (one per scene, never rebuilt) ---
        img = self._make_image()
        self._img = img
        pm = QPixmap.fromImage(img)
        pi = QGraphicsPixmapItem(pm)
        pi.setPos(pr.topLeft())
        pi.setTransformationMode(Qt.FastTransformation)
        # Scale from (n_residues × n_steps) pixels to plot_rect size
        pi.setTransform(
            pi.transform().scale(
                pr.width() / max(self._n_res_shown(), 1),
                pr.height() / max(run.n_steps, 1),
            )
        )
        self.addItem(pi)
        self._pixmap_item = pi

        # --- Overlays ---
        step_pen = QPen(_STEP_OVL_COLOR, 1.5)
        step_pen.setCosmetic(True)
        sl = QGraphicsLineItem(pr.left(), pr.top(), pr.right(), pr.top())
        sl.setPen(step_pen)
        sl.setZValue(20)
        self.addItem(sl)
        self._step_line = sl

        res_pen = QPen(_RES_OVL_COLOR, 1.5)
        res_pen.setCosmetic(True)
        rl = QGraphicsLineItem(pr.left(), pr.top(), pr.left(), pr.bottom())
        rl.setPen(res_pen)
        rl.setZValue(20)
        rl.setVisible(False)
        self.addItem(rl)
        self._res_line = rl

        self._rebuild_axes()
        self._rebuild_legend()
        self._update_scene_rect()

    def _make_image(self) -> QImage:
        run = self._run
        col_lo = self._res_lo
        col_hi = self._res_hi + 1
        if self._mode == "absolute":
            data = run.plddt_matrix[:, col_lo:col_hi]
            pixels = alphafold_color_array(data)
        else:
            data = run.plddt_delta[:, col_lo:col_hi]
            vmax = float(np.nanmax(np.abs(run.plddt_delta))) or 30.0
            pixels = delta_color_array(data, vmax)

        n_steps, n_res = pixels.shape
        raw = pixels.astype(np.uint32).tobytes()
        # QImage(data, w, h, bytes_per_line, format) — raw must stay alive until .copy()
        img = QImage(raw, n_res, n_steps, n_res * 4, QImage.Format_ARGB32)
        return img.copy()  # detach from `raw` buffer

    def _refill_image(self) -> None:
        if self._pixmap_item is None:
            return
        pr = self._plot_rect()
        img = self._make_image()
        self._img = img
        self._pixmap_item.setPixmap(QPixmap.fromImage(img))
        # Re-apply scale for new residue range / same step count
        t = self._pixmap_item.transform()
        t.reset()
        n_res = self._n_res_shown()
        t.scale(
            pr.width() / max(n_res, 1),
            pr.height() / max(self._run.n_steps, 1),
        )
        self._pixmap_item.setTransform(t)

    def _rebuild_axes(self) -> None:
        for item in self._axis_items:
            self.removeItem(item)
        self._axis_items = []

        run = self._run
        pr = self._plot_rect()
        ax_pen = QPen(QColor(80, 80, 80), 1)
        ax_pen.setCosmetic(True)
        tick_pen = QPen(QColor(80, 80, 80), 1)
        tick_pen.setCosmetic(True)
        lbl_color = QColor(50, 50, 50)

        def add(item: QGraphicsItem) -> None:
            item.setZValue(15)
            self.addItem(item)
            self._axis_items.append(item)

        # Border
        border = QGraphicsRectItem(pr)
        border.setPen(ax_pen)
        border.setBrush(Qt.NoBrush)
        add(border)

        # Y axis — steps; aim for ~10 visible labels
        step_ticks = nice_ticks(0, run.n_steps - 1, min(10, run.n_steps))
        for s in step_ticks:
            y = self._step_to_scene_y(s)
            tl = QGraphicsLineItem(pr.left() - 4, y, pr.left(), y)
            tl.setPen(tick_pen)
            add(tl)
            lbl = self.addText(f"{int(s)}")
            lbl.setDefaultTextColor(lbl_color)
            font = lbl.font()
            font.setPointSize(7)
            lbl.setFont(font)
            br = lbl.boundingRect()
            lbl.setPos(pr.left() - 6 - br.width(), y - br.height() / 2)
            lbl.setZValue(15)
            self._axis_items.append(lbl)

        # X axis — residue indices
        n_res = self._n_res_shown()
        density_target = max(4, min(20, n_res // 15))
        res_ticks = nice_ticks(self._res_lo, self._res_hi, density_target)
        for r in res_ticks:
            x = self._res_to_scene_x(r)
            tl = QGraphicsLineItem(x, pr.bottom(), x, pr.bottom() + 4)
            tl.setPen(tick_pen)
            add(tl)
            lbl = self.addText(f"{int(r)}")
            lbl.setDefaultTextColor(lbl_color)
            font = lbl.font()
            font.setPointSize(7)
            lbl.setFont(font)
            br = lbl.boundingRect()
            lbl.setPos(x - br.width() / 2, pr.bottom() + 6)
            lbl.setZValue(15)
            self._axis_items.append(lbl)

        # Axis labels
        y_lbl = self.addText("Step")
        y_lbl.setDefaultTextColor(lbl_color)
        y_lbl.setZValue(15)
        y_lbl.setRotation(-90)
        y_lbl.setPos(pr.left() - 40, pr.center().y() + y_lbl.boundingRect().width() / 2)
        self._axis_items.append(y_lbl)

        x_lbl = self.addText("Residue")
        x_lbl.setDefaultTextColor(lbl_color)
        x_lbl.setZValue(15)
        x_lbl.setPos(pr.center().x() - x_lbl.boundingRect().width() / 2, pr.bottom() + 16)
        self._axis_items.append(x_lbl)

    def _rebuild_legend(self) -> None:
        for item in self._legend_items:
            self.removeItem(item)
        self._legend_items = []

        pr = self._plot_rect()
        lx = pr.right() + _LEGEND_GAP
        ly = pr.top()
        lw = _LEGEND_W
        lh = _LEGEND_H

        if self._mode == "absolute":
            bands = [
                (QColor(0, 83, 214),   "≥90"),
                (QColor(101, 203, 243), "70–89"),
                (QColor(255, 219, 19),  "50–69"),
                (QColor(255, 125, 69),  "<50"),
            ]
            bh = lh / len(bands)
            for i, (color, label) in enumerate(bands):
                rect = QGraphicsRectItem(lx, ly + i * bh, lw, bh)
                rect.setBrush(color)
                rect.setPen(Qt.NoPen)
                rect.setZValue(15)
                self.addItem(rect)
                self._legend_items.append(rect)
                t = self.addText(label)
                t.setDefaultTextColor(QColor(50, 50, 50))
                f = t.font(); f.setPointSize(7); t.setFont(f)
                t.setPos(lx + lw + 3, ly + i * bh + bh / 2 - t.boundingRect().height() / 2)
                t.setZValue(15)
                self._legend_items.append(t)
        else:
            # Gradient strip: red (top) → white (mid) → blue (bottom)
            grad_vals = np.linspace(1.0, -1.0, int(lh))
            for i, t_val in enumerate(grad_vals):
                if t_val >= 0:
                    r = int(255 + t_val * (38 - 255))
                    g = int(255 + t_val * (139 - 255))
                    b = int(255 + t_val * (210 - 255))
                else:
                    frac = -t_val
                    r = int(255 + frac * (220 - 255))
                    g = int(255 + frac * (50 - 255))
                    b = int(255 + frac * (47 - 255))
                rect = QGraphicsRectItem(lx, ly + i, lw, 1.5)
                rect.setBrush(QColor(r, g, b))
                rect.setPen(Qt.NoPen)
                rect.setZValue(15)
                self.addItem(rect)
                self._legend_items.append(rect)

            vmax = float(np.nanmax(np.abs(self._run.plddt_delta))) or 30.0
            for label, frac in [(f"+{vmax:.0f}", 0.0), ("0", 0.5), (f"−{vmax:.0f}", 1.0)]:
                t = self.addText(label)
                t.setDefaultTextColor(QColor(50, 50, 50))
                f = t.font(); f.setPointSize(7); t.setFont(f)
                t.setPos(lx + lw + 3, ly + frac * lh - t.boundingRect().height() / 2)
                t.setZValue(15)
                self._legend_items.append(t)

    def _update_scene_rect(self) -> None:
        self.setSceneRect(self.itemsBoundingRect().adjusted(-4, -4, 4, 4))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _plot_rect(self) -> QRectF:
        # Fixed scene layout — wide enough for a typical run
        w = 600.0
        h = 400.0
        return QRectF(_LEFT_MARGIN, _TOP_MARGIN, w, h)

    def _n_res_shown(self) -> int:
        return max(1, self._res_hi - self._res_lo + 1)

    def _step_to_scene_y(self, step: float) -> float:
        pr = self._plot_rect()
        n = self._run.n_steps
        return pr.top() + (step / max(n - 1, 1)) * pr.height()

    def _res_to_scene_x(self, residue: float) -> float:
        pr = self._plot_rect()
        n = self._n_res_shown()
        return pr.left() + ((residue - self._res_lo) / max(n - 1, 1)) * pr.width()

    def scene_to_step_res(self, scene_pos: QPointF) -> tuple[int, int]:
        pr = self._plot_rect()
        frac_y = (scene_pos.y() - pr.top()) / pr.height()
        frac_x = (scene_pos.x() - pr.left()) / pr.width()
        step = int(round(frac_y * (self._run.n_steps - 1)))
        res = int(round(frac_x * (self._n_res_shown() - 1))) + self._res_lo
        step = max(0, min(self._run.n_steps - 1, step))
        res = max(self._res_lo, min(self._res_hi, res))
        return step, res


class HeatmapView(QGraphicsView):
    def __init__(self, run: PtttRun, ctrl: SelectionController, parent=None) -> None:
        self._scene = HeatmapScene(run)
        super().__init__(self._scene, parent)
        self._ctrl = ctrl
        self._run = run

        # OpenGL viewport (matches template pattern)
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        gl = QOpenGLWidget()
        gl.setFormat(fmt)
        self.setViewport(gl)

        self.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QColor(245, 245, 245))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)

        # Connect controller signals → scene updates
        ctrl.currentStepChanged.connect(self._scene.move_step_line)
        ctrl.residueSelectedChanged.connect(self._scene.move_res_line)

    # ------------------------------------------------------------------
    # Public API (called from MainWindow)
    # ------------------------------------------------------------------

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._scene.set_run(run)

    def set_color_mode(self, mode: str) -> None:
        self._scene.set_color_mode(mode)  # type: ignore[arg-type]

    def set_residue_range(self, lo: int, hi: int) -> None:
        self._scene.set_residue_range(lo, hi)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        pr = self._scene._plot_rect()
        if pr.contains(sp):
            step, res = self._scene.scene_to_step_res(sp)
            self._ctrl.setHoveredResidue(res)
            plddt_val = self._run.plddt_matrix[step, res]
            QToolTip.showText(
                event.globalPos(),
                f"Step {step} · Res {res}\npLDDT = {plddt_val:.1f}",
            )
        else:
            self._ctrl.setHoveredResidue(-1)
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        pr = self._scene._plot_rect()
        if pr.contains(sp):
            step, res = self._scene.scene_to_step_res(sp)
            if event.modifiers() & Qt.ShiftModifier:
                self._ctrl.toggleComparisonStep(step)
            else:
                self._ctrl.setCurrentStep(step)
                self._ctrl.setSelectedResidue(res)
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1 + event.angleDelta().y() * 0.001
        self.scale(factor, factor)
