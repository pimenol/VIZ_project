"""Axis drawing helpers"""

import math
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsScene, QGraphicsTextItem


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    if hi <= lo or target < 2:
        return [lo, hi]
    span = hi - lo
    raw_step = span / (target - 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    if norm <= 1.0:
        nice = 1.0
    elif norm <= 2.0:
        nice = 2.0
    elif norm <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * mag
    start = math.ceil(lo / step) * step
    ticks: list[float] = []
    v = start
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 10))
        v += step
    return ticks


@dataclass
class AxisItems:
    x_ticks: list[QGraphicsLineItem] = field(default_factory=list)
    y_ticks: list[QGraphicsLineItem] = field(default_factory=list)
    x_labels: list[QGraphicsTextItem] = field(default_factory=list)
    y_labels: list[QGraphicsTextItem] = field(default_factory=list)
    x_gridlines: list[QGraphicsLineItem] = field(default_factory=list)
    y_gridlines: list[QGraphicsLineItem] = field(default_factory=list)
    border: list[QGraphicsLineItem] = field(default_factory=list)


_TICK_LEN = 5.0
_LABEL_OFFSET = 4.0
_GRID_COLOR = QColor(220, 220, 220)
_AXIS_COLOR = QColor(80, 80, 80)
_LABEL_COLOR = QColor(60, 60, 60)


def draw_axes(
    scene: QGraphicsScene,
    plot_rect: QRectF,
    x_ticks: list[float],
    y_ticks: list[float],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    x_fmt: Callable[[float], str] = lambda v: f"{v:g}",
    y_fmt: Callable[[float], str] = lambda v: f"{v:g}",
    draw_grid: bool = True,
    font_size: int = 8,
    z: float = 10.0,
) -> AxisItems:
    ax_pen = QPen(_AXIS_COLOR, 1.0)
    ax_pen.setCosmetic(True)
    tick_pen = QPen(_AXIS_COLOR, 1.0)
    tick_pen.setCosmetic(True)
    grid_pen = QPen(_GRID_COLOR, 0.5, Qt.DashLine)
    grid_pen.setCosmetic(True)

    items = AxisItems()
    px = plot_rect.x()
    py = plot_rect.y()
    pw = plot_rect.width()
    ph = plot_rect.height()

    def x_scene(v: float) -> float:
        return px + (v - x_lo) / (x_hi - x_lo) * pw if x_hi != x_lo else px

    def y_scene(v: float) -> float:
        return py + ph - (v - y_lo) / (y_hi - y_lo) * ph if y_hi != y_lo else py + ph

    for line_coords in [
        (px, py + ph, px + pw, py + ph),
        (px, py, px, py + ph),
    ]:
        ln = QGraphicsLineItem(*line_coords)
        ln.setPen(ax_pen)
        ln.setZValue(z)
        scene.addItem(ln)
        items.border.append(ln)

    bottom = py + ph
    for v in x_ticks:
        sx = x_scene(v)
        tl = QGraphicsLineItem(sx, bottom, sx, bottom + _TICK_LEN)
        tl.setPen(tick_pen)
        tl.setZValue(z)
        scene.addItem(tl)
        items.x_ticks.append(tl)
        if draw_grid:
            gl = QGraphicsLineItem(sx, py, sx, bottom)
            gl.setPen(grid_pen)
            gl.setZValue(z - 1)
            scene.addItem(gl)
            items.x_gridlines.append(gl)
        lbl = scene.addText(x_fmt(v))
        lbl.setDefaultTextColor(_LABEL_COLOR)
        lbl.setZValue(z)
        font = lbl.font()
        font.setPointSize(font_size)
        lbl.setFont(font)
        br = lbl.boundingRect()
        lbl.setPos(sx - br.width() / 2, bottom + _TICK_LEN + _LABEL_OFFSET)
        items.x_labels.append(lbl)

    left = px
    for v in y_ticks:
        sy = y_scene(v)
        tl = QGraphicsLineItem(left - _TICK_LEN, sy, left, sy)
        tl.setPen(tick_pen)
        tl.setZValue(z)
        scene.addItem(tl)
        items.y_ticks.append(tl)
        if draw_grid:
            gl = QGraphicsLineItem(left, sy, left + pw, sy)
            gl.setPen(grid_pen)
            gl.setZValue(z - 1)
            scene.addItem(gl)
            items.y_gridlines.append(gl)
        lbl = scene.addText(y_fmt(v))
        lbl.setDefaultTextColor(_LABEL_COLOR)
        lbl.setZValue(z)
        font = lbl.font()
        font.setPointSize(font_size)
        lbl.setFont(font)
        br = lbl.boundingRect()
        lbl.setPos(left - _TICK_LEN - _LABEL_OFFSET - br.width(), sy - br.height() / 2)
        items.y_labels.append(lbl)

    return items
