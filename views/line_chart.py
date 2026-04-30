"""T1 — multi-panel line chart of global metrics across TTT steps."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QSurfaceFormat,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QToolTip,
)

from chart_axes import nice_ticks
from controller import SelectionController
from data import PtttRun

# Panel layout (in scene coords)
_LEFT = 60.0
_RIGHT_MARGIN = 20.0
_TOP = 10.0
_GAP = 14.0
_BOT_MARGIN = 28.0
_TOTAL_W = 700.0
_TOTAL_H = 480.0

# Relative panel heights: pLDDT 40%, then 20% each
_PANEL_FRACS = [0.40, 0.20, 0.20, 0.20]
_PANEL_TITLES = ["pLDDT", "Perplexity", "TM-score", "lDDT"]
_PANEL_COLORS = [
    QColor(0, 83, 214),    # blue — pLDDT
    QColor(200, 80, 40),   # orange-red — perplexity
    QColor(80, 160, 80),   # green — tm_score
    QColor(140, 40, 160),  # purple — lDDT
]


def _series_for_panel(run: PtttRun, idx: int) -> np.ndarray:
    return [run.plddt_mean, run.perplexity, run.tm_score, run.lddt][idx]


def _panel_rects(total_h: float) -> list[QRectF]:
    """Return the data-plot QRectF for each of the 4 panels."""
    plot_h = total_h - _TOP - _BOT_MARGIN - _GAP * 3
    rects = []
    y = _TOP
    for frac in _PANEL_FRACS:
        h = plot_h * frac
        rects.append(QRectF(_LEFT, y, _TOTAL_W - _LEFT - _RIGHT_MARGIN, h))
        y += h + _GAP
    return rects


class LineChartScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self._run = run
        self._step_line: QGraphicsLineItem | None = None
        self._crosshair_x: QGraphicsLineItem | None = None
        self._crosshair_labels: list[QGraphicsTextItem] = []
        self._panel_items: list[list[QGraphicsItem]] = [[] for _ in range(4)]
        self._build()

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self.clear()
        self._step_line = None
        self._crosshair_x = None
        self._crosshair_labels = []
        self._panel_items = [[] for _ in range(4)]
        self._build()

    def move_step_line(self, step: int) -> None:
        if self._step_line is None:
            return
        rects = _panel_rects(_TOTAL_H)
        x = self._step_to_x(step, rects[0])
        top = rects[0].top()
        bot = rects[-1].bottom()
        self._step_line.setLine(x, top, x, bot)

    def show_crosshair(self, step: int, scene_x: float) -> None:
        if self._crosshair_x is None:
            return
        rects = _panel_rects(_TOTAL_H)
        top = rects[0].top()
        bot = rects[-1].bottom()
        self._crosshair_x.setLine(scene_x, top, scene_x, bot)
        self._crosshair_x.setVisible(True)

        run = self._run
        series_vals = [
            run.plddt_mean[step] if step < run.n_steps else math.nan,
            run.perplexity[step] if step < run.n_steps else math.nan,
            run.tm_score[step] if step < run.n_steps else math.nan,
            run.lddt[step] if step < run.n_steps else math.nan,
        ]
        labels = [
            f"pLDDT={series_vals[0]:.2f}" if not math.isnan(series_vals[0]) else "pLDDT=—",
            f"PPL={series_vals[1]:.2f}" if not math.isnan(series_vals[1]) else "PPL=—",
            f"TM={series_vals[2]:.3f}" if not math.isnan(series_vals[2]) else "TM=—",
            f"lDDT={series_vals[3]:.3f}" if not math.isnan(series_vals[3]) else "lDDT=—",
        ]
        for i, (lbl_item, lbl_text, rect) in enumerate(
            zip(self._crosshair_labels, labels, rects)
        ):
            lbl_item.setPlainText(lbl_text)
            lbl_item.setPos(scene_x + 4, rect.top() + 2)
            lbl_item.setVisible(True)

    def hide_crosshair(self) -> None:
        if self._crosshair_x:
            self._crosshair_x.setVisible(False)
        for lbl in self._crosshair_labels:
            lbl.setVisible(False)

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        run = self._run
        rects = _panel_rects(_TOTAL_H)

        for i, (rect, title, color) in enumerate(
            zip(rects, _PANEL_TITLES, _PANEL_COLORS)
        ):
            self._draw_panel(i, rect, title, color)

        # Shared X axis label
        xr = rects[-1]
        lbl = self.addText("Step")
        lbl.setDefaultTextColor(QColor(60, 60, 60))
        lbl.setPos(
            _LEFT + (xr.width() - lbl.boundingRect().width()) / 2,
            xr.bottom() + 16,
        )
        lbl.setZValue(5)

        # Current-step vertical line
        pen = QPen(QColor(255, 100, 0), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        sl = QGraphicsLineItem(_LEFT, rects[0].top(), _LEFT, rects[-1].bottom())
        sl.setPen(pen)
        sl.setZValue(18)
        self.addItem(sl)
        self._step_line = sl

        # Crosshair (hidden by default)
        ch_pen = QPen(QColor(120, 120, 120), 1, Qt.DotLine)
        ch_pen.setCosmetic(True)
        ch = QGraphicsLineItem(_LEFT, rects[0].top(), _LEFT, rects[-1].bottom())
        ch.setPen(ch_pen)
        ch.setZValue(17)
        ch.setVisible(False)
        self.addItem(ch)
        self._crosshair_x = ch

        self._crosshair_labels = []
        for rect in rects:
            t = self.addText("")
            t.setDefaultTextColor(QColor(60, 60, 60))
            font = t.font()
            font.setPointSize(7)
            t.setFont(font)
            t.setZValue(17)
            t.setVisible(False)
            self._crosshair_labels.append(t)

        self.setSceneRect(self.itemsBoundingRect().adjusted(-4, -4, 4, 4))

    def _draw_panel(self, idx: int, rect: QRectF, title: str, color: QColor) -> None:
        run = self._run
        series = _series_for_panel(run, idx)
        all_nan = np.all(np.isnan(series))

        # Background
        bg = QGraphicsRectItem(rect)
        bg.setBrush(QColor(250, 250, 250))
        bg.setPen(QPen(QColor(200, 200, 200), 0.5))
        bg.setZValue(1)
        self.addItem(bg)

        # Title
        t = self.addText(title)
        t.setDefaultTextColor(QColor(80, 80, 80))
        font = t.font()
        font.setPointSize(8)
        font.setBold(True)
        t.setFont(font)
        t.setPos(rect.left() + 4, rect.top() + 2)
        t.setZValue(10)

        if all_nan:
            nd = self.addText("no data")
            nd.setDefaultTextColor(QColor(160, 160, 160))
            nd.setPos(
                rect.center().x() - nd.boundingRect().width() / 2,
                rect.center().y() - nd.boundingRect().height() / 2,
            )
            nd.setZValue(10)
            self._draw_x_axis(rect, run.n_steps - 1, idx == 3)
            return

        y_lo = float(np.nanmin(series))
        y_hi = float(np.nanmax(series))
        if y_lo == y_hi:
            y_lo -= 1; y_hi += 1

        # Y axis ticks + grid
        y_ticks = nice_ticks(y_lo, y_hi, 4)
        ax_pen = QPen(QColor(80, 80, 80), 0.8)
        ax_pen.setCosmetic(True)
        tick_pen = ax_pen
        grid_pen = QPen(QColor(220, 220, 220), 0.5, Qt.DashLine)
        grid_pen.setCosmetic(True)
        lbl_color = QColor(60, 60, 60)

        for v in y_ticks:
            sy = self._val_to_y(v, y_lo, y_hi, rect)
            tl = QGraphicsLineItem(rect.left() - 4, sy, rect.left(), sy)
            tl.setPen(tick_pen)
            tl.setZValue(5)
            self.addItem(tl)
            gl = QGraphicsLineItem(rect.left(), sy, rect.right(), sy)
            gl.setPen(grid_pen)
            gl.setZValue(2)
            self.addItem(gl)
            lbl = self.addText(f"{v:g}")
            lbl.setDefaultTextColor(lbl_color)
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            br = lbl.boundingRect()
            lbl.setPos(rect.left() - 6 - br.width(), sy - br.height() / 2)
            lbl.setZValue(5)

        # Border
        border = QGraphicsLineItem(rect.left(), rect.top(), rect.left(), rect.bottom())
        border.setPen(ax_pen)
        border.setZValue(6)
        self.addItem(border)
        hline = QGraphicsLineItem(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        hline.setPen(ax_pen)
        hline.setZValue(6)
        self.addItem(hline)

        # Data curve
        path = QPainterPath()
        started = False
        for s_int in range(run.n_steps):
            v = series[s_int]
            if math.isnan(v):
                started = False
                continue
            sx = self._step_to_x(s_int, rect)
            sy = self._val_to_y(v, y_lo, y_hi, rect)
            if not started:
                path.moveTo(sx, sy)
                started = True
            else:
                path.lineTo(sx, sy)

        curve_pen = QPen(color, 1.8)
        curve_pen.setCosmetic(True)
        curve_item = QGraphicsPathItem(path)
        curve_item.setPen(curve_pen)
        curve_item.setZValue(8)
        self.addItem(curve_item)

        # Peak pLDDT marker (panel 0 only)
        if idx == 0:
            bs = run.best_step
            bv = run.best_plddt
            bx = self._step_to_x(bs, rect)
            by = self._val_to_y(bv, y_lo, y_hi, rect)
            dot = QGraphicsEllipseItem(bx - 4, by - 4, 8, 8)
            dot.setBrush(QColor(255, 60, 60))
            dot.setPen(QPen(Qt.white, 1))
            dot.setZValue(12)
            self.addItem(dot)
            peak_lbl = self.addText(f"peak\nstep {bs}")
            peak_lbl.setDefaultTextColor(QColor(180, 0, 0))
            f = peak_lbl.font(); f.setPointSize(7); peak_lbl.setFont(f)
            peak_lbl.setPos(bx + 6, by - peak_lbl.boundingRect().height() / 2)
            peak_lbl.setZValue(12)

        self._draw_x_axis(rect, run.n_steps - 1, idx == 3)

    def _draw_x_axis(self, rect: QRectF, max_step: int, show_labels: bool) -> None:
        ax_pen = QPen(QColor(80, 80, 80), 0.8)
        ax_pen.setCosmetic(True)
        tick_pen = ax_pen
        x_ticks = nice_ticks(0, max_step, 6)
        for v in x_ticks:
            sx = self._step_to_x(v, rect)
            tl = QGraphicsLineItem(sx, rect.bottom(), sx, rect.bottom() + 4)
            tl.setPen(tick_pen)
            tl.setZValue(5)
            self.addItem(tl)
            if show_labels:
                lbl = self.addText(f"{int(v)}")
                lbl.setDefaultTextColor(QColor(60, 60, 60))
                f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
                br = lbl.boundingRect()
                lbl.setPos(sx - br.width() / 2, rect.bottom() + 5)
                lbl.setZValue(5)

    def _step_to_x(self, step: float, rect: QRectF) -> float:
        n = self._run.n_steps
        return rect.left() + (step / max(n - 1, 1)) * rect.width()

    def _val_to_y(self, val: float, lo: float, hi: float, rect: QRectF) -> float:
        return rect.bottom() - (val - lo) / (hi - lo) * rect.height()

    def nearest_step(self, scene_x: float, rect: QRectF) -> int:
        frac = (scene_x - rect.left()) / rect.width()
        s = int(round(frac * (self._run.n_steps - 1)))
        return max(0, min(self._run.n_steps - 1, s))

    def in_any_panel(self, scene_pos: QPointF) -> bool:
        for rect in _panel_rects(_TOTAL_H):
            if rect.contains(scene_pos):
                return True
        return False

    def first_rect(self) -> QRectF:
        return _panel_rects(_TOTAL_H)[0]


class LineChartView(QGraphicsView):
    def __init__(self, run: PtttRun, ctrl: SelectionController, parent=None) -> None:
        self._lscene = LineChartScene(run)
        super().__init__(self._lscene, parent)
        self._ctrl = ctrl
        self._run = run
        self._dragging = False
        self._drag_start_x = 0.0

        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        gl = QOpenGLWidget()
        gl.setFormat(fmt)
        self.setViewport(gl)

        self.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QColor(240, 240, 240))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        ctrl.currentStepChanged.connect(self._lscene.move_step_line)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._lscene.set_run(run)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        if self._lscene.in_any_panel(sp):
            rect = self._lscene.first_rect()
            step = self._lscene.nearest_step(sp.x(), rect)
            self._lscene.show_crosshair(step, self._lscene._step_to_x(step, rect))
            vals = [
                self._run.plddt_mean[step],
                self._run.perplexity[step],
                self._run.tm_score[step],
                self._run.lddt[step],
            ]
            def fmt(v):
                return f"{v:.3f}" if not math.isnan(v) else "—"
            QToolTip.showText(
                event.globalPos(),
                f"Step {step}\npLDDT={fmt(vals[0])}  PPL={fmt(vals[1])}\nTM={fmt(vals[2])}  lDDT={fmt(vals[3])}",
            )
        else:
            self._lscene.hide_crosshair()
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        if self._lscene.in_any_panel(sp):
            rect = self._lscene.first_rect()
            step = self._lscene.nearest_step(sp.x(), rect)
            self._ctrl.setCurrentStep(step)
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:
        self._lscene.hide_crosshair()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1 + event.angleDelta().y() * 0.001
        self.scale(factor, 1.0)  # X-only zoom
