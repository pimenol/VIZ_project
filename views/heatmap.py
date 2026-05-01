"""T2 - per-residue pLDDT heatmap backed by a single QImage."""

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
    QGraphicsView,
    QToolTip,
)

from chart_axes import nice_ticks
from colors import alphafold_color_array, delta_color_array
from controller import SelectionController
from data import PtttRun, ss_segments

_LEFT_MARGIN = 52.0
_TOP_MARGIN = 10.0
_PLOT_W = 600.0
_PLOT_H = 400.0
_PLOT_RECT = QRectF(_LEFT_MARGIN, _TOP_MARGIN, _PLOT_W, _PLOT_H)

_LEGEND_W = 18.0
_LEGEND_H = 120.0
_LEGEND_GAP = 8.0

# _DELTA_NEG = (220, 50, 47)  
# _DELTA_MID = (255, 255, 255)
# _DELTA_POS = (38, 139, 210) 

_STEP_OVL_COLOR = QColor(255, 80, 0)
_RES_OVL_COLOR = QColor(0, 100, 220)


def _delta_vmax(plddt_delta: np.ndarray) -> float:
    return float(np.nanmax(np.abs(plddt_delta))) or 30.0


def _make_gradient_pixmap(height: int, vmax: float) -> QPixmap:
    deltas = np.linspace(vmax, -vmax, height, dtype=np.float32).reshape(-1, 1)
    pixels = np.ascontiguousarray(delta_color_array(deltas, vmax), dtype=np.uint32)
    img = QImage(pixels, 1, height, 4, QImage.Format_ARGB32)
    return QPixmap.fromImage(img.copy())


class HeatmapScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self._run = run
        self._mode: Literal["absolute", "delta"] = "absolute"
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._delta_vmax_cached = _delta_vmax(run.plddt_delta)
        self._img_buf: np.ndarray | None = None
        self._img: QImage | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._step_line: QGraphicsLineItem | None = None
        self._res_line: QGraphicsLineItem | None = None
        self._axis_items: list[QGraphicsItem] = []
        self._legend_items: list[QGraphicsItem] = []
        self._ss_filter: set[int] = {0, 1, 2}
        self._ss_filter_step: int = 0
        self._ss_overlay_items: list[QGraphicsRectItem] = []
        self._build()

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._delta_vmax_cached = _delta_vmax(run.plddt_delta)
        self._img_buf = None
        self._img = None
        self._pixmap_item = None
        self._step_line = None
        self._res_line = None
        self._axis_items = []
        self._legend_items = []
        self._ss_overlay_items = []
        self.clear()
        self._build()
        self._rebuild_ss_overlay()

    def set_color_mode(self, mode: Literal["absolute", "delta"]) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._refill_image()
        self._rebuild_legend()

    def set_residue_range(self, lo: int, hi: int) -> None:
        if (lo, hi) == (self._res_lo, self._res_hi):
            return
        self._res_lo = lo
        self._res_hi = hi
        self._refill_image()
        self._rebuild_axes()
        self._rebuild_ss_overlay()

    def set_ss_filter(self, allowed: set[int]) -> None:
        self._ss_filter = set(allowed)
        self._rebuild_ss_overlay()

    def set_ss_filter_step(self, step: int) -> None:
        if step == self._ss_filter_step or step < 0 or step >= self._run.n_steps:
            return
        self._ss_filter_step = step
        self._rebuild_ss_overlay()

    def _rebuild_ss_overlay(self) -> None:
        for item in self._ss_overlay_items:
            self.removeItem(item)
        self._ss_overlay_items = []
        if self._ss_filter == {0, 1, 2}:
            return
        ss_row = self._run.ss_matrix[self._ss_filter_step]
        sub = ss_row[self._res_lo : self._res_hi + 1]
        overlay_brush = QColor(255, 255, 255, 180)
        overlay_pen = QPen(Qt.NoPen)
        for lo_local, hi_local, label in ss_segments(sub):
            if int(label) in self._ss_filter:
                continue
            res_lo_abs = lo_local + self._res_lo
            res_hi_abs = hi_local + self._res_lo
            x_left = self._res_to_scene_x(float(res_lo_abs) - 0.5)
            x_right = self._res_to_scene_x(float(res_hi_abs) + 0.5)
            x_left = max(_PLOT_RECT.left(), x_left)
            x_right = min(_PLOT_RECT.right(), x_right)
            if x_right <= x_left:
                continue
            rect = QGraphicsRectItem(x_left, _PLOT_RECT.top(), x_right - x_left, _PLOT_RECT.height())
            rect.setBrush(overlay_brush)
            rect.setPen(overlay_pen)
            rect.setZValue(18) 
            self.addItem(rect)
            self._ss_overlay_items.append(rect)

    def move_step_line(self, step: int) -> None:
        if self._step_line is None:
            return
        y = self._step_to_scene_y(step)
        self._step_line.setLine(_PLOT_RECT.left(), y, _PLOT_RECT.right(), y)

    def move_res_line(self, residue: int) -> None:
        if self._res_line is None:
            return
        if residue < 0:
            self._res_line.setVisible(False)
            return
        self._res_line.setVisible(True)
        x = self._res_to_scene_x(residue)
        self._res_line.setLine(x, _PLOT_RECT.top(), x, _PLOT_RECT.bottom())

    def _build(self) -> None:
        pr = _PLOT_RECT
        img = self._make_image()
        pi = QGraphicsPixmapItem(QPixmap.fromImage(img))
        pi.setPos(pr.topLeft())
        pi.setTransformationMode(Qt.FastTransformation)
        self._apply_pixmap_scale(pi)
        self.addItem(pi)
        self._pixmap_item = pi

        step_pen = QPen(_STEP_OVL_COLOR, 1.5); step_pen.setCosmetic(True)
        sl = QGraphicsLineItem(pr.left(), pr.top(), pr.right(), pr.top())
        sl.setPen(step_pen); sl.setZValue(20)
        self.addItem(sl)
        self._step_line = sl

        res_pen = QPen(_RES_OVL_COLOR, 1.5); res_pen.setCosmetic(True)
        rl = QGraphicsLineItem(pr.left(), pr.top(), pr.left(), pr.bottom())
        rl.setPen(res_pen); rl.setZValue(20); rl.setVisible(False)
        self.addItem(rl)
        self._res_line = rl

        self._rebuild_axes()
        self._rebuild_legend()
        self.setSceneRect(self.itemsBoundingRect().adjusted(-4, -4, 4, 4))

    def _apply_pixmap_scale(self, item: QGraphicsPixmapItem) -> None:
        t = item.transform()
        t.reset()
        t.scale(
            _PLOT_W / max(self._n_res_shown(), 1),
            _PLOT_H / max(self._run.n_steps, 1),
        )
        item.setTransform(t)

    def _make_image(self) -> QImage:
        run = self._run
        col_lo = self._res_lo
        col_hi = self._res_hi + 1
        if self._mode == "absolute":
            pixels = alphafold_color_array(run.plddt_matrix[:, col_lo:col_hi])
        else:
            pixels = delta_color_array(
                run.plddt_delta[:, col_lo:col_hi], self._delta_vmax_cached
            )

        self._img_buf = np.ascontiguousarray(pixels, dtype=np.uint32)
        n_steps, n_res = self._img_buf.shape
        self._img = QImage(
            self._img_buf, n_res, n_steps, n_res * 4, QImage.Format_ARGB32
        )
        return self._img

    def _refill_image(self) -> None:
        if self._pixmap_item is None:
            return
        self._pixmap_item.setPixmap(QPixmap.fromImage(self._make_image()))
        self._apply_pixmap_scale(self._pixmap_item)

    def _rebuild_axes(self) -> None:
        for item in self._axis_items:
            self.removeItem(item)
        self._axis_items = []

        run = self._run
        pr = _PLOT_RECT
        ax_pen = QPen(QColor(80, 80, 80), 1); ax_pen.setCosmetic(True)
        lbl_color = QColor(50, 50, 50)

        def add_new(item: QGraphicsItem) -> None:
            item.setZValue(15)
            self.addItem(item)
            self._axis_items.append(item)

        def track(item: QGraphicsItem) -> None:
            item.setZValue(15)
            self._axis_items.append(item)

        def add_tick_label(text: str, anchor_x: float, anchor_y: float, align: str) -> None:
            lbl = self.addText(text)
            lbl.setDefaultTextColor(lbl_color)
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            br = lbl.boundingRect()
            if align == "right":
                lbl.setPos(anchor_x - br.width(), anchor_y - br.height() / 2)
            else:  # "center-below"
                lbl.setPos(anchor_x - br.width() / 2, anchor_y)
            track(lbl)

        border = QGraphicsRectItem(pr)
        border.setPen(ax_pen); border.setBrush(Qt.NoBrush)
        add_new(border)

        for s in nice_ticks(0, run.n_steps - 1, min(10, run.n_steps)):
            y = self._step_to_scene_y(s)
            tl = QGraphicsLineItem(pr.left() - 4, y, pr.left(), y)
            tl.setPen(ax_pen)
            add_new(tl)
            add_tick_label(f"{int(s)}", pr.left() - 6, y, "right")

        n_res = self._n_res_shown()
        density_target = max(4, min(20, n_res // 15))
        for r in nice_ticks(self._res_lo, self._res_hi, density_target):
            x = self._res_to_scene_x(r)
            tl = QGraphicsLineItem(x, pr.bottom(), x, pr.bottom() + 4)
            tl.setPen(ax_pen)
            add_new(tl)
            add_tick_label(f"{int(r)}", x, pr.bottom() + 6, "center-below")

        y_lbl = self.addText("Step")
        y_lbl.setDefaultTextColor(lbl_color)
        y_lbl.setRotation(-90)
        y_lbl.setPos(pr.left() - 40, pr.center().y() + y_lbl.boundingRect().width() / 2)
        track(y_lbl)

        x_lbl = self.addText("Residue")
        x_lbl.setDefaultTextColor(lbl_color)
        x_lbl.setPos(pr.center().x() - x_lbl.boundingRect().width() / 2, pr.bottom() + 16)
        track(x_lbl)

    def _rebuild_legend(self) -> None:
        for item in self._legend_items:
            self.removeItem(item)
        self._legend_items = []

        pr = _PLOT_RECT
        lx = pr.right() + _LEGEND_GAP
        ly = pr.top()
        lw = _LEGEND_W
        lh = _LEGEND_H
        lbl_color = QColor(50, 50, 50)

        def add_label(text: str, x: float, y_center: float) -> None:
            t = self.addText(text)
            t.setDefaultTextColor(lbl_color)
            f = t.font(); f.setPointSize(7); t.setFont(f)
            t.setPos(x, y_center - t.boundingRect().height() / 2)
            t.setZValue(15)
            self._legend_items.append(t)

        if self._mode == "absolute":
            bands = [
                (QColor(0, 83, 214),    "≥90"),
                (QColor(101, 203, 243), "70–89"),
                (QColor(255, 219, 19),  "50–69"),
                (QColor(255, 125, 69),  "<50"),
            ]
            bh = lh / len(bands)
            for i, (color, label) in enumerate(bands):
                rect = QGraphicsRectItem(lx, ly + i * bh, lw, bh)
                rect.setBrush(color); rect.setPen(Qt.NoPen); rect.setZValue(15)
                self.addItem(rect)
                self._legend_items.append(rect)
                add_label(label, lx + lw + 3, ly + i * bh + bh / 2)
        else:
            vmax = self._delta_vmax_cached
            pi = QGraphicsPixmapItem(_make_gradient_pixmap(int(lh), vmax))
            pi.setPos(lx, ly)
            t = pi.transform(); t.reset(); t.scale(lw, 1.0); pi.setTransform(t)
            pi.setZValue(15)
            self.addItem(pi)
            self._legend_items.append(pi)

            for text, frac in ((f"+{vmax:.0f}", 0.0), ("0", 0.5), (f"−{vmax:.0f}", 1.0)):
                add_label(text, lx + lw + 3, ly + frac * lh)

    def _n_res_shown(self) -> int:
        return max(1, self._res_hi - self._res_lo + 1)

    def _step_to_scene_y(self, step: float) -> float:
        return _PLOT_RECT.top() + (step / max(self._run.n_steps - 1, 1)) * _PLOT_H

    def _res_to_scene_x(self, residue: float) -> float:
        n = self._n_res_shown()
        return _PLOT_RECT.left() + ((residue - self._res_lo) / max(n - 1, 1)) * _PLOT_W

    def scene_to_step_res(self, scene_pos: QPointF) -> tuple[int, int]:
        pr = _PLOT_RECT
        frac_y = (scene_pos.y() - pr.top()) / _PLOT_H
        frac_x = (scene_pos.x() - pr.left()) / _PLOT_W
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

        ctrl.currentStepChanged.connect(self._scene.move_step_line)
        ctrl.currentStepChanged.connect(self._scene.set_ss_filter_step)
        ctrl.residueSelectedChanged.connect(self._scene.move_res_line)
        ctrl.ssClassFilterChanged.connect(self._scene.set_ss_filter)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._scene.set_run(run)

    def set_color_mode(self, mode: str) -> None:
        self._scene.set_color_mode(mode) 

    def set_residue_range(self, lo: int, hi: int) -> None:
        self._scene.set_residue_range(lo, hi)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        if _PLOT_RECT.contains(sp):
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
        if _PLOT_RECT.contains(sp):
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
