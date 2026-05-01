import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGraphicsItem

_FADED_ALPHA = 0x40   


class PointsItem(QGraphicsItem):
    def __init__(
        self,
        coords: np.ndarray,
        colors_argb: np.ndarray,
        radius: float = 4.0,
    ) -> None:
        super().__init__()
        self._radius = float(radius)
        self._coords: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._colors: np.ndarray = np.empty((0,), dtype=np.uint32)
        self._alpha_mask: np.ndarray | None = None
        self._groups: list[tuple[int, np.ndarray]] = []
        self._bounding_rect = QRectF()
        self.set_data(coords, colors_argb)

    def set_data(self, coords: np.ndarray, colors_argb: np.ndarray) -> None:
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        colors = np.ascontiguousarray(colors_argb, dtype=np.uint32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got {coords.shape}")
        if colors.shape != (coords.shape[0],):
            raise ValueError(f"colors must be (N,), got {colors.shape}")
        self.prepareGeometryChange()
        self._coords = coords
        self._colors = colors
        if self._alpha_mask is not None and self._alpha_mask.shape != (coords.shape[0],):
            self._alpha_mask = None
        self._rebuild_groups()
        self._bounding_rect = self._compute_bounds()
        self.update()

    def set_alpha_mask(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self._alpha_mask = None
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self._coords.shape[0],):
                raise ValueError(f"mask must be ({self._coords.shape[0]},), got {mask.shape}")
            self._alpha_mask = mask
        self._rebuild_groups()
        self.update()

    def index_at(self, scene_point: QPointF, max_dist_px: float) -> int:
        if self._coords.size == 0:
            return -1
        dx = self._coords[:, 0] - scene_point.x()
        dy = self._coords[:, 1] - scene_point.y()
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        if d2[i] > max_dist_px * max_dist_px:
            return -1
        return i

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        r = self._radius
        d = r * 2.0
        for argb, pts in self._groups:
            painter.setBrush(QColor.fromRgba(argb))
            for x, y in pts:
                painter.drawEllipse(QRectF(x - r, y - r, d, d))

    def _rebuild_groups(self) -> None:
        if self._coords.size == 0:
            self._groups = []
            return
        if self._alpha_mask is None:
            effective = self._colors
        else:
            faded = (self._colors & np.uint32(0x00FFFFFF)) | np.uint32(_FADED_ALPHA << 24)
            effective = np.where(self._alpha_mask, self._colors, faded).astype(np.uint32)
        unique, inverse = np.unique(effective, return_inverse=True)
        self._groups = [
            (int(unique[i]), self._coords[inverse == i])
            for i in range(unique.size)
        ]

    def _compute_bounds(self) -> QRectF:
        if self._coords.size == 0:
            return QRectF()
        xs = self._coords[:, 0]
        ys = self._coords[:, 1]
        r = self._radius
        return QRectF(
            float(xs.min()) - r,
            float(ys.min()) - r,
            float(xs.max() - xs.min()) + 2 * r,
            float(ys.max() - ys.min()) + 2 * r,
        )
