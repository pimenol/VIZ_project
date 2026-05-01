"""T1 — multi-panel line chart of global metrics across TTT steps."""

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
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QToolTip,
)

from chart_axes import nice_ticks
from colors import SS_COLORS, SS_NAMES
from controller import SelectionController
from data import PtttRun

_LEFT = 60.0
_RIGHT_MARGIN = 20.0
_TOP = 10.0
_GAP = 14.0
_BOT_MARGIN = 28.0
_TOTAL_W = 700.0
_TOTAL_H = 480.0

_PANEL_FRACS = [0.60, 0.40]
_PANEL_TITLES = ["pLDDT", "lDDT"]
_PANEL_COLORS = [QColor(0, 83, 214), QColor(140, 40, 160)]
_PANEL_PRECISIONS = [".2f", ".3f"]


def _build_panel_rects() -> list[QRectF]:
    n_gaps = max(0, len(_PANEL_FRACS) - 1)
    plot_h = _TOTAL_H - _TOP - _BOT_MARGIN - _GAP * n_gaps
    width = _TOTAL_W - _LEFT - _RIGHT_MARGIN
    rects = []
    y = _TOP
    for frac in _PANEL_FRACS:
        h = plot_h * frac
        rects.append(QRectF(_LEFT, y, width, h))
        y += h + _GAP
    return rects


_RECTS = _build_panel_rects()
_BOTTOM_PANEL = len(_PANEL_FRACS) - 1


def _series_for(run: PtttRun, idx: int) -> np.ndarray:
    return (run.plddt_mean, run.lddt)[idx]


def _ss_stratified_plddt_means(run: PtttRun) -> np.ndarray:
    """Per-step mean pLDDT for residues whose SS label matches each class.

    Returns shape (3, S): row 0 = helix, 1 = sheet, 2 = coil. NaN where no residues match.
    SS changes across steps, so the mask is recomputed per row of ss_matrix.
    """
    plddt = run.plddt_matrix.astype(np.float64)
    ss = run.ss_matrix
    out = np.full((3, run.n_steps), np.nan, dtype=np.float64)
    for label in range(3):
        mask = ss == label
        counts = mask.sum(axis=1)
        sums = (plddt * mask).sum(axis=1)
        valid = counts > 0
        out[label, valid] = sums[valid] / counts[valid]
    return out


class LineChartScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self._run = run
        self._step_line: QGraphicsLineItem | None = None
        self._crosshair_x: QGraphicsLineItem | None = None
        self._crosshair_labels: list[QGraphicsTextItem] = []
        self._ss_visible = False
        self._ss_path_items: list[QGraphicsPathItem] = []
        self._ss_legend_items: list = []
        self._build()

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self.clear()
        self._step_line = None
        self._crosshair_x = None
        self._crosshair_labels = []
        self._ss_path_items = []
        self._ss_legend_items = []
        self._build()
        self._apply_ss_visibility()

    def set_ss_visible(self, visible: bool) -> None:
        if visible == self._ss_visible:
            return
        self._ss_visible = visible
        self._apply_ss_visibility()

    def _apply_ss_visibility(self) -> None:
        for item in self._ss_path_items:
            item.setVisible(self._ss_visible)
        for item in self._ss_legend_items:
            item.setVisible(self._ss_visible)

    def move_step_line(self, step: int) -> None:
        if self._step_line is None:
            return
        x = self._step_to_x(step, _RECTS[0])
        self._step_line.setLine(x, _RECTS[0].top(), x, _RECTS[-1].bottom())

    def show_crosshair(self, step: int, scene_x: float) -> None:
        if self._crosshair_x is None:
            return
        self._crosshair_x.setLine(scene_x, _RECTS[0].top(), scene_x, _RECTS[-1].bottom())
        self._crosshair_x.setVisible(True)

        run = self._run
        in_range = step < run.n_steps
        for i, (lbl_item, name, prec, rect) in enumerate(
            zip(self._crosshair_labels, _PANEL_TITLES, _PANEL_PRECISIONS, _RECTS)
        ):
            v = _series_for(run, i)[step] if in_range else math.nan
            text = f"{name}=—" if math.isnan(v) else f"{name}={v:{prec}}"
            lbl_item.setPlainText(text)
            lbl_item.setPos(scene_x + 4, rect.top() + 2)
            lbl_item.setVisible(True)

    def hide_crosshair(self) -> None:
        if self._crosshair_x:
            self._crosshair_x.setVisible(False)
        for lbl in self._crosshair_labels:
            lbl.setVisible(False)

    def _build(self) -> None:
        for i, (rect, title, color) in enumerate(
            zip(_RECTS, _PANEL_TITLES, _PANEL_COLORS)
        ):
            self._draw_panel(i, rect, title, color)

        xr = _RECTS[-1]
        lbl = self.addText("Step")
        lbl.setDefaultTextColor(QColor(60, 60, 60))
        lbl.setPos(
            _LEFT + (xr.width() - lbl.boundingRect().width()) / 2,
            xr.bottom() + 16,
        )
        lbl.setZValue(5)

        top, bot = _RECTS[0].top(), _RECTS[-1].bottom()
        step_pen = QPen(QColor(255, 100, 0), 1.5, Qt.DashLine)
        step_pen.setCosmetic(True)
        sl = QGraphicsLineItem(_LEFT, top, _LEFT, bot)
        sl.setPen(step_pen)
        sl.setZValue(18)
        self.addItem(sl)
        self._step_line = sl

        ch_pen = QPen(QColor(120, 120, 120), 1, Qt.DotLine)
        ch_pen.setCosmetic(True)
        ch = QGraphicsLineItem(_LEFT, top, _LEFT, bot)
        ch.setPen(ch_pen)
        ch.setZValue(17)
        ch.setVisible(False)
        self.addItem(ch)
        self._crosshair_x = ch

        self._crosshair_labels = []
        for _ in _RECTS:
            t = self.addText("")
            t.setDefaultTextColor(QColor(60, 60, 60))
            font = t.font(); font.setPointSize(7); t.setFont(font)
            t.setZValue(17)
            t.setVisible(False)
            self._crosshair_labels.append(t)

        self.setSceneRect(self.itemsBoundingRect().adjusted(-4, -4, 4, 4))

    def _draw_panel(self, idx: int, rect: QRectF, title: str, color: QColor) -> None:
        run = self._run
        series = _series_for(run, idx)
        is_bottom = idx == _BOTTOM_PANEL

        bg = QGraphicsRectItem(rect)
        bg.setBrush(QColor(250, 250, 250))
        bg.setPen(QPen(QColor(200, 200, 200), 0.5))
        bg.setZValue(1)
        self.addItem(bg)

        t = self.addText(title)
        t.setDefaultTextColor(QColor(80, 80, 80))
        font = t.font(); font.setPointSize(8); font.setBold(True); t.setFont(font)
        t.setPos(rect.left() + 4, rect.top() + 2)
        t.setZValue(10)

        if np.all(np.isnan(series)):
            nd = self.addText("no data")
            nd.setDefaultTextColor(QColor(160, 160, 160))
            nd.setPos(
                rect.center().x() - nd.boundingRect().width() / 2,
                rect.center().y() - nd.boundingRect().height() / 2,
            )
            nd.setZValue(10)
            self._draw_x_axis(rect, run.n_steps - 1, is_bottom)
            return

        y_lo = float(np.nanmin(series))
        y_hi = float(np.nanmax(series))
        # On the pLDDT panel, expand range to cover SS-stratified means even when hidden,
        # so the y axis stays stable when the user toggles them on.
        ss_strat: np.ndarray | None = None
        if idx == 0:
            ss_strat = _ss_stratified_plddt_means(run)
            finite = ss_strat[np.isfinite(ss_strat)]
            if finite.size:
                y_lo = min(y_lo, float(finite.min()))
                y_hi = max(y_hi, float(finite.max()))
        if y_lo == y_hi:
            y_lo -= 1; y_hi += 1

        ax_pen = QPen(QColor(80, 80, 80), 0.8); ax_pen.setCosmetic(True)
        grid_pen = QPen(QColor(220, 220, 220), 0.5, Qt.DashLine); grid_pen.setCosmetic(True)
        lbl_color = QColor(60, 60, 60)

        for v in nice_ticks(y_lo, y_hi, 4):
            sy = self._val_to_y(v, y_lo, y_hi, rect)
            tl = QGraphicsLineItem(rect.left() - 4, sy, rect.left(), sy)
            tl.setPen(ax_pen); tl.setZValue(5)
            self.addItem(tl)
            gl = QGraphicsLineItem(rect.left(), sy, rect.right(), sy)
            gl.setPen(grid_pen); gl.setZValue(2)
            self.addItem(gl)
            lbl = self.addText(f"{v:g}")
            lbl.setDefaultTextColor(lbl_color)
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            br = lbl.boundingRect()
            lbl.setPos(rect.left() - 6 - br.width(), sy - br.height() / 2)
            lbl.setZValue(5)

        for x1, y1, x2, y2 in [
            (rect.left(),  rect.top(),    rect.left(),  rect.bottom()),
            (rect.left(),  rect.bottom(), rect.right(), rect.bottom()),
        ]:
            ln = QGraphicsLineItem(x1, y1, x2, y2)
            ln.setPen(ax_pen); ln.setZValue(6)
            self.addItem(ln)

        path = QPainterPath()
        started = False
        for s_int in range(run.n_steps):
            v = series[s_int]
            if math.isnan(v):
                started = False
                continue
            sx = self._step_to_x(s_int, rect)
            sy = self._val_to_y(v, y_lo, y_hi, rect)
            if started:
                path.lineTo(sx, sy)
            else:
                path.moveTo(sx, sy)
                started = True

        curve_pen = QPen(color, 1.8); curve_pen.setCosmetic(True)
        curve_item = QGraphicsPathItem(path)
        curve_item.setPen(curve_pen); curve_item.setZValue(8)
        self.addItem(curve_item)

        if idx == 0:
            bs, bv = run.best_step, run.best_plddt
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

            if ss_strat is not None:
                self._draw_ss_stratified(rect, ss_strat, y_lo, y_hi)

        self._draw_x_axis(rect, run.n_steps - 1, is_bottom)

    def _draw_ss_stratified(
        self, rect: QRectF, ss_strat: np.ndarray, y_lo: float, y_hi: float
    ) -> None:
        for label in range(3):
            series = ss_strat[label]
            path = QPainterPath()
            started = False
            for s_int in range(series.size):
                v = series[s_int]
                if not np.isfinite(v):
                    started = False
                    continue
                sx = self._step_to_x(s_int, rect)
                sy = self._val_to_y(float(v), y_lo, y_hi, rect)
                if started:
                    path.lineTo(sx, sy)
                else:
                    path.moveTo(sx, sy)
                    started = True
            pen = QPen(SS_COLORS[label], 1.4, Qt.DashLine)
            pen.setCosmetic(True)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(7)  # below the main pLDDT curve (z=8)
            item.setVisible(self._ss_visible)
            self.addItem(item)
            self._ss_path_items.append(item)

        # Legend in the panel's top-right (under the title at top-left).
        sw = 10.0
        gap = 4.0
        font_pt = 7
        x = rect.right() - 4.0
        y = rect.top() + 4.0
        # Build right-to-left so we can right-align without knowing widths up front.
        for label in (2, 1, 0):
            name = SS_NAMES[label]
            txt = self.addText(name)
            txt.setDefaultTextColor(QColor(60, 60, 60))
            f = txt.font(); f.setPointSize(font_pt); txt.setFont(f)
            br = txt.boundingRect()
            txt.setPos(x - br.width(), y - 2.0)
            txt.setZValue(11)
            txt.setVisible(self._ss_visible)
            self._ss_legend_items.append(txt)
            x -= br.width() + 2.0

            rect_item = QGraphicsRectItem(x - sw, y, sw, sw)
            rect_item.setBrush(SS_COLORS[label])
            rect_item.setPen(QPen(QColor(80, 80, 80), 0.4))
            rect_item.setZValue(11)
            rect_item.setVisible(self._ss_visible)
            self.addItem(rect_item)
            self._ss_legend_items.append(rect_item)
            x -= sw + gap

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
        return any(rect.contains(scene_pos) for rect in _RECTS)

    def first_rect(self) -> QRectF:
        return _RECTS[0]


class LineChartView(QGraphicsView):
    def __init__(self, run: PtttRun, ctrl: SelectionController, parent=None) -> None:
        self._lscene = LineChartScene(run)
        super().__init__(self._lscene, parent)
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
        self.setBackgroundBrush(QColor(240, 240, 240))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        ctrl.currentStepChanged.connect(self._lscene.move_step_line)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._lscene.set_run(run)

    def set_ss_visible(self, visible: bool) -> None:
        self._lscene.set_ss_visible(visible)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        if self._lscene.in_any_panel(sp):
            rect = self._lscene.first_rect()
            step = self._lscene.nearest_step(sp.x(), rect)
            self._lscene.show_crosshair(step, self._lscene._step_to_x(step, rect))
            run = self._run
            parts = [f"Step {step}"]
            for name, ser in zip(_PANEL_TITLES, (run.plddt_mean, run.lddt)):
                v = ser[step]
                parts.append(f"{name}={'—' if math.isnan(v) else f'{v:.3f}'}")
            QToolTip.showText(event.globalPos(), "  ".join(parts))
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
