"""T3 — overlaid per-residue pLDDT profiles + step picker."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QSurfaceFormat,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from chart_axes import nice_ticks
from controller import SelectionController
from data import PtttRun, ss_segments
from views.ss_track import SecondaryStructureTrack

# Layout
_LEFT = 52.0
_RIGHT_MARGIN = 130.0   # room for legend
_TOP = 10.0
_BOT_MARGIN = 28.0
_TOTAL_W = 560.0
_TOTAL_H = 380.0

_Y_LO = 0.0
_Y_HI = 100.0
_REF_LEVELS = (50.0, 70.0, 90.0)

_LEGEND_GAP_X = 8.0
_LEGEND_ROW_H = 16.0
_LEGEND_SWATCH_W = 12.0
_LEGEND_SWATCH_H = 10.0

# 8-color categorical palette (Tableau-inspired, no red/gray so they don't clash with overlays)
_PALETTE = [
    QColor(31, 119, 180),   # blue
    QColor(255, 127, 14),   # orange
    QColor(44, 160, 44),    # green
    QColor(148, 103, 189),  # purple
    QColor(140, 86, 75),    # brown
    QColor(23, 190, 207),   # cyan
    QColor(188, 189, 34),   # olive
    QColor(214, 39, 40),    # red (last resort)
]


_PLOT_RECT = QRectF(_LEFT, _TOP, _TOTAL_W - _LEFT - _RIGHT_MARGIN, _TOTAL_H - _TOP - _BOT_MARGIN)


def _step_color(idx: int) -> QColor:
    return _PALETTE[idx % len(_PALETTE)]


class ProfileScene(QGraphicsScene):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self._run = run
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._comparison: list[int] = []
        self._step_paths: dict[int, QGraphicsPathItem] = {}
        self._res_line: QGraphicsLineItem | None = None
        self._static_items_built = False
        self._ss_filter: set[int] = {0, 1, 2}
        self._ss_filter_step: int = 0
        self._ss_overlay_items: list[QGraphicsRectItem] = []
        self._build_static()

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._comparison = []
        # Reset item-ref lists BEFORE clear() so _rebuild_axes won't try to
        # removeItem() already-deleted C++ objects.
        self._step_paths = {}
        self._res_line = None
        self._static_items_built = False
        self._x_axis_items = []
        self._legend_items = []
        self._ss_overlay_items = []
        self.clear()
        self._build_static()
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
        pr = _PLOT_RECT
        for lo_local, hi_local, label in ss_segments(sub):
            if int(label) in self._ss_filter:
                continue
            res_lo_abs = lo_local + self._res_lo
            res_hi_abs = hi_local + self._res_lo
            x_left = self._res_to_x(float(res_lo_abs) - 0.5, pr)
            x_right = self._res_to_x(float(res_hi_abs) + 0.5, pr)
            x_left = max(pr.left(), x_left)
            x_right = min(pr.right(), x_right)
            if x_right <= x_left:
                continue
            rect = QGraphicsRectItem(x_left, pr.top(), x_right - x_left, pr.height())
            rect.setBrush(overlay_brush)
            rect.setPen(overlay_pen)
            rect.setZValue(18)
            self.addItem(rect)
            self._ss_overlay_items.append(rect)

    def set_residue_range(self, lo: int, hi: int) -> None:
        if (lo, hi) == (self._res_lo, self._res_hi):
            return
        self._res_lo = lo
        self._res_hi = hi
        self._rebuild_axes()
        for step, item in self._step_paths.items():
            item.setPath(self._build_path(step))
        self._rebuild_ss_overlay()

    def set_comparison(self, steps: list[int]) -> None:
        sorted_steps = sorted(set(steps))
        idx_of = {s: i for i, s in enumerate(sorted_steps)}
        old = set(self._comparison)
        new = set(sorted_steps)
        for s in old - new:
            if s in self._step_paths:
                self.removeItem(self._step_paths.pop(s))
        for s in sorted(new - old):
            self._add_path(s, idx_of[s])
        for s, item in self._step_paths.items():
            item.setPen(self._step_pen(idx_of[s], s))
        self._comparison = sorted_steps
        self._rebuild_legend(sorted_steps, idx_of)

    def _step_pen(self, idx: int, step: int) -> QPen:
        width = 1.5 + (1.0 if step == self._run.best_step else 0.0)
        pen = QPen(_step_color(idx), width)
        pen.setCosmetic(True)
        return pen

    def move_res_line(self, residue: int) -> None:
        if self._res_line is None:
            return
        if residue < 0:
            self._res_line.setVisible(False)
            return
        self._res_line.setVisible(True)
        x = self._res_to_x(residue, _PLOT_RECT)
        self._res_line.setLine(x, _PLOT_RECT.top(), x, _PLOT_RECT.bottom())

    def _build_static(self) -> None:
        pr = _PLOT_RECT
        ax_pen = QPen(QColor(80, 80, 80), 0.8); ax_pen.setCosmetic(True)
        grid_pen = QPen(QColor(220, 220, 220), 0.5, Qt.DashLine); grid_pen.setCosmetic(True)
        ref_pen = QPen(QColor(150, 150, 150), 0.8, Qt.DashLine); ref_pen.setCosmetic(True)
        lbl_color = QColor(60, 60, 60)

        bg = QGraphicsRectItem(pr)
        bg.setBrush(QColor(250, 250, 250))
        bg.setPen(QPen(QColor(200, 200, 200), 0.5))
        bg.setZValue(1)
        self.addItem(bg)

        for ref_val in _REF_LEVELS:
            y = self._val_to_y(ref_val, pr)
            rl = QGraphicsLineItem(pr.left(), y, pr.right(), y)
            rl.setPen(ref_pen); rl.setZValue(3)
            self.addItem(rl)
            lbl = self.addText(f"{int(ref_val)}")
            lbl.setDefaultTextColor(QColor(160, 160, 160))
            f = lbl.font(); f.setPointSize(6); lbl.setFont(f)
            lbl.setPos(pr.right() + 3, y - lbl.boundingRect().height() / 2)
            lbl.setZValue(3)

        for v in nice_ticks(_Y_LO, _Y_HI, 5):
            y = self._val_to_y(v, pr)
            tl = QGraphicsLineItem(pr.left() - 4, y, pr.left(), y)
            tl.setPen(ax_pen); tl.setZValue(5)
            self.addItem(tl)
            gl = QGraphicsLineItem(pr.left(), y, pr.right(), y)
            gl.setPen(grid_pen); gl.setZValue(2)
            self.addItem(gl)
            lbl = self.addText(f"{v:g}")
            lbl.setDefaultTextColor(lbl_color)
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            br = lbl.boundingRect()
            lbl.setPos(pr.left() - 6 - br.width(), y - br.height() / 2)
            lbl.setZValue(5)

        for coords in [
            (pr.left(), pr.bottom(), pr.right(), pr.bottom()),
            (pr.left(), pr.top(),    pr.left(),  pr.bottom()),
        ]:
            ln = QGraphicsLineItem(*coords)
            ln.setPen(ax_pen); ln.setZValue(6)
            self.addItem(ln)

        yl = self.addText("pLDDT")
        yl.setDefaultTextColor(lbl_color); yl.setZValue(5); yl.setRotation(-90)
        yl.setPos(pr.left() - 40, pr.center().y() + yl.boundingRect().width() / 2)

        xl = self.addText("Residue")
        xl.setDefaultTextColor(lbl_color); xl.setZValue(5)
        xl.setPos(pr.center().x() - xl.boundingRect().width() / 2, pr.bottom() + 14)

        self._rebuild_axes()

        res_pen = QPen(QColor(0, 100, 220), 1.5); res_pen.setCosmetic(True)
        rl = QGraphicsLineItem(pr.left(), pr.top(), pr.left(), pr.bottom())
        rl.setPen(res_pen); rl.setZValue(20); rl.setVisible(False)
        self.addItem(rl)
        self._res_line = rl

        self._legend_items = []
        self._static_items_built = True
        self.setSceneRect(self.itemsBoundingRect().adjusted(-4, -4, 4, 4))

    def _rebuild_axes(self) -> None:
        for item in getattr(self, "_x_axis_items", []):
            self.removeItem(item)
        self._x_axis_items = []

        pr = _PLOT_RECT
        ax_pen = QPen(QColor(80, 80, 80), 0.8); ax_pen.setCosmetic(True)
        lbl_color = QColor(60, 60, 60)
        n = self._res_hi - self._res_lo + 1
        density_target = max(4, min(16, n // 10))
        for v in nice_ticks(self._res_lo, self._res_hi, density_target):
            x = self._res_to_x(v, pr)
            tl = QGraphicsLineItem(x, pr.bottom(), x, pr.bottom() + 4)
            tl.setPen(ax_pen); tl.setZValue(5)
            self.addItem(tl)
            self._x_axis_items.append(tl)
            lbl = self.addText(f"{int(v)}")
            lbl.setDefaultTextColor(lbl_color)
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            br = lbl.boundingRect()
            lbl.setPos(x - br.width() / 2, pr.bottom() + 5)
            lbl.setZValue(5)
            self._x_axis_items.append(lbl)

    def _build_path(self, step: int) -> QPainterPath:
        pr = _PLOT_RECT
        plddt = self._run.plddt_matrix[step]
        path = QPainterPath()
        for i, res in enumerate(range(self._res_lo, self._res_hi + 1)):
            x = self._res_to_x(res, pr)
            y = self._val_to_y(float(plddt[res]), pr)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        return path

    def _add_path(self, step: int, idx: int) -> None:
        item = QGraphicsPathItem(self._build_path(step))
        item.setPen(self._step_pen(idx, step))
        item.setZValue(10)
        self.addItem(item)
        self._step_paths[step] = item

    def _rebuild_legend(self, sorted_steps: list[int], idx_of: dict[int, int]) -> None:
        for item in self._legend_items:
            self.removeItem(item)
        self._legend_items = []

        pr = _PLOT_RECT
        lx = pr.right() + _LEGEND_GAP_X
        ly = pr.top()

        for i, step in enumerate(sorted_steps):
            sw = QGraphicsRectItem(lx, ly + i * _LEGEND_ROW_H, _LEGEND_SWATCH_W, _LEGEND_SWATCH_H)
            sw.setBrush(_step_color(idx_of[step]))
            sw.setPen(Qt.NoPen); sw.setZValue(15)
            self.addItem(sw)
            self._legend_items.append(sw)

            label = f"Step {step}"
            if step == self._run.best_step:
                label += " ★"
            lbl = self.addText(label)
            lbl.setDefaultTextColor(QColor(50, 50, 50))
            f = lbl.font(); f.setPointSize(7); lbl.setFont(f)
            lbl.setPos(lx + _LEGEND_SWATCH_W + 4, ly + i * _LEGEND_ROW_H - 2)
            lbl.setZValue(15)
            self._legend_items.append(lbl)

    def _res_to_x(self, res: float, pr: QRectF) -> float:
        n = max(1, self._res_hi - self._res_lo)
        return pr.left() + (res - self._res_lo) / n * pr.width()

    def _val_to_y(self, val: float, pr: QRectF) -> float:
        return pr.bottom() - (val - _Y_LO) / (_Y_HI - _Y_LO) * pr.height()

    def x_to_residue(self, scene_x: float) -> int:
        pr = _PLOT_RECT
        frac = (scene_x - pr.left()) / pr.width()
        r = int(round(frac * (self._res_hi - self._res_lo) + self._res_lo))
        return max(self._res_lo, min(self._res_hi, r))

    def in_plot(self, x: float, y: float) -> bool:
        return _PLOT_RECT.contains(x, y)


class ProfileView(QWidget):
    """Compound widget: QGraphicsView (left) + step picker QListWidget (right)."""

    def __init__(self, run: PtttRun, ctrl: SelectionController, parent=None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._run = run
        self._updating_list = False  # guard against recursive updates

        self._pscene = ProfileScene(run)

        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        gl = QOpenGLWidget()
        gl.setFormat(fmt)

        self._gview = QGraphicsView(self._pscene)
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

        self._picker = QListWidget()
        self._picker.setFixedWidth(110)
        self._picker.itemChanged.connect(self._on_list_item_changed)
        self._populate_picker(run)

        self._ss_track = SecondaryStructureTrack(
            run, ctrl,
            plot_left=_PLOT_RECT.left(),
            plot_width=_PLOT_RECT.width(),
            parent=self,
        )

        plot_col = QWidget(self)
        v = QVBoxLayout(plot_col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._ss_track)
        v.addWidget(self._gview)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(plot_col, 1)
        layout.addWidget(self._picker)

        ctrl.comparisonStepsChanged.connect(self._on_comparison_changed)
        ctrl.residueSelectedChanged.connect(self._pscene.move_res_line)
        ctrl.currentStepChanged.connect(self._pscene.set_ss_filter_step)
        ctrl.ssClassFilterChanged.connect(self._pscene.set_ss_filter)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._pscene.set_run(run)
        self._populate_picker(run)
        self._ss_track.set_run(run)

    def set_residue_range(self, lo: int, hi: int) -> None:
        self._pscene.set_residue_range(lo, hi)
        self._ss_track.set_residue_range(lo, hi)

    def _populate_picker(self, run: PtttRun) -> None:
        self._picker.blockSignals(True)
        self._picker.clear()
        for s in range(run.n_steps):
            item = QListWidgetItem(f"Step {s}")
            item.setData(Qt.UserRole, s)
            item.setCheckState(Qt.Unchecked)
            self._picker.addItem(item)
        self._picker.blockSignals(False)

    def _on_comparison_changed(self, steps: list[int]) -> None:
        self._pscene.set_comparison(steps)
        self._updating_list = True
        self._picker.blockSignals(True)
        for i in range(self._picker.count()):
            item = self._picker.item(i)
            s = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if s in steps else Qt.Unchecked)
        self._picker.blockSignals(False)
        self._updating_list = False

    def _on_list_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_list:
            return
        step = item.data(Qt.UserRole)
        self._ctrl.toggleComparisonStep(step)

    def _view_mouse_move(self, event) -> None:
        sp = self._gview.mapToScene(event.pos())
        if self._pscene.in_plot(sp.x(), sp.y()):
            res = self._pscene.x_to_residue(sp.x())
            self._ctrl.setHoveredResidue(res)
            tips = []
            for step in sorted(self._ctrl.comparison_steps):
                v = float(self._run.plddt_matrix[step, res])
                tips.append(f"Step {step}: {v:.1f}")
            QToolTip.showText(event.globalPos(), f"Res {res}\n" + "\n".join(tips))
        else:
            self._ctrl.setHoveredResidue(-1)
            QToolTip.hideText()
        QGraphicsView.mouseMoveEvent(self._gview, event)

    def _view_mouse_press(self, event) -> None:
        sp = self._gview.mapToScene(event.pos())
        if self._pscene.in_plot(sp.x(), sp.y()):
            res = self._pscene.x_to_residue(sp.x())
            self._ctrl.setSelectedResidue(res)
        QGraphicsView.mousePressEvent(self._gview, event)
