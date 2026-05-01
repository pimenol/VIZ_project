"""Phase B — secondary-structure annotation strip aligned to a residue X axis."""

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QToolTip,
)

# Allow `python views/ss_track.py`-style imports during dev — same pattern as embedding_view.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colors import SS_COLORS, SS_LETTERS, SS_NAMES, ss_color
from controller import SelectionController
from data import PtttRun, ss_segments

_TRACK_HEIGHT = 30.0          # widget pixel height
_PLOT_TOP = 4.0               # vertical inset from top
_PLOT_BOTTOM_INSET = 4.0      # vertical inset from bottom
_LABEL_MIN_PX = 20.0          # only draw "H/E/C" label if segment >= this px wide
_RECT_Z = 5
_LABEL_Z = 6


class SecondaryStructureTrack(QGraphicsView):
    """Thin horizontal strip of colored segments showing SS labels for the current step.

    Aligns with the parent view's plot region via `plot_left` and `plot_width`. Updates on
    `currentStepChanged` and on residue-range changes propagated through `set_residue_range`.
    """

    def __init__(
        self,
        run: PtttRun,
        ctrl: SelectionController,
        plot_left: float,
        plot_width: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._run = run
        self._ctrl = ctrl
        self._plot_left = plot_left
        self._plot_width = plot_width
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._step = 0
        self._segments: list[tuple[int, int, int]] = []  # populated lazily

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor(248, 248, 248))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setFixedHeight(int(_TRACK_HEIGHT))
        self.setMouseTracking(True)

        self._segments_group = QGraphicsItemGroup()
        self._scene.addItem(self._segments_group)
        # Children of segments_group are recreated on every refresh.

        self._refresh()
        ctrl.currentStepChanged.connect(self.set_step)

    def set_run(self, run: PtttRun) -> None:
        self._run = run
        self._res_lo = 0
        self._res_hi = run.n_residues - 1
        self._step = 0
        self._refresh()

    def set_step(self, step: int) -> None:
        if step == self._step or step < 0 or step >= self._run.n_steps:
            return
        self._step = step
        self._refresh()

    def set_residue_range(self, lo: int, hi: int) -> None:
        if (lo, hi) == (self._res_lo, self._res_hi):
            return
        self._res_lo = lo
        self._res_hi = hi
        self._refresh()

    def set_plot_geometry(self, plot_left: float, plot_width: float) -> None:
        self._plot_left = plot_left
        self._plot_width = plot_width
        self._refresh()

    def resizeEvent(self, event):  # noqa: D401 — Qt override
        super().resizeEvent(event)
        # Keep scene rect in sync with view width so coordinate mapping below is well-defined.
        self._scene.setSceneRect(QRectF(0, 0, self.viewport().width(), _TRACK_HEIGHT))
        self._refresh()

    def mouseMoveEvent(self, event):  # noqa: D401 — Qt override
        sp_x = self.mapToScene(event.pos()).x()
        residue = self._scene_x_to_residue(sp_x)
        if residue < 0:
            self._ctrl.setHoveredResidue(-1)
            QToolTip.hideText()
        else:
            seg = self._segment_at_residue(residue)
            self._ctrl.setHoveredResidue(residue)
            if seg is not None:
                lo, hi, label = seg
                QToolTip.showText(
                    event.globalPos(),
                    f"{SS_NAMES[label]} · residues {lo}–{hi} · length {hi - lo + 1}",
                )
        super().mouseMoveEvent(event)

    def _segment_at_residue(self, residue: int) -> tuple[int, int, int] | None:
        for lo, hi, label in self._segments:
            if lo <= residue <= hi:
                return (lo, hi, label)
        return None

    def _scene_x_to_residue(self, scene_x: float) -> int:
        n = self._res_hi - self._res_lo + 1
        if n <= 0 or self._plot_width <= 0:
            return -1
        frac = (scene_x - self._plot_left) / self._plot_width
        if frac < 0 or frac > 1:
            return -1
        return int(round(self._res_lo + frac * (n - 1)))

    def _residue_edges_to_scene_x(self, residue: int) -> tuple[float, float]:
        """Return (x_left, x_right) of a residue cell in scene coords (inclusive boundaries)."""
        n = self._res_hi - self._res_lo + 1
        cell_w = self._plot_width / max(n, 1)
        x_left = self._plot_left + (residue - self._res_lo) * cell_w
        return x_left, x_left + cell_w

    def _refresh(self) -> None:
        # Children of the group are owned by it; removing the group's children clears them.
        for child in list(self._segments_group.childItems()):
            self._segments_group.removeFromGroup(child)
            self._scene.removeItem(child)

        ss_row = self._run.ss_matrix[self._step]
        # Restrict to residue range; ss_segments operates on the slice, then offset back.
        sub = ss_row[self._res_lo : self._res_hi + 1]
        local_segs = ss_segments(sub)
        self._segments = [
            (lo + self._res_lo, hi + self._res_lo, label) for lo, hi, label in local_segs
        ]

        rect_top = _PLOT_TOP
        rect_h = _TRACK_HEIGHT - _PLOT_TOP - _PLOT_BOTTOM_INSET
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)

        for lo, hi, label in self._segments:
            x_left, _ = self._residue_edges_to_scene_x(lo)
            _, x_right = self._residue_edges_to_scene_x(hi)
            width = max(1.0, x_right - x_left)
            rect_item = QGraphicsRectItem(x_left, rect_top, width, rect_h)
            color = ss_color(label)
            rect_item.setBrush(QBrush(color))
            pen = QPen(color.darker(140), 0.5)
            pen.setCosmetic(True)
            rect_item.setPen(pen)
            rect_item.setZValue(_RECT_Z)
            self._segments_group.addToGroup(rect_item)

            if width >= _LABEL_MIN_PX:
                txt = QGraphicsSimpleTextItem(SS_LETTERS[label])
                txt.setFont(font)
                txt.setBrush(QBrush(QColor(35, 35, 35)))
                br = txt.boundingRect()
                txt.setPos(
                    x_left + (width - br.width()) / 2,
                    rect_top + (rect_h - br.height()) / 2,
                )
                txt.setZValue(_LABEL_Z)
                self._segments_group.addToGroup(txt)
