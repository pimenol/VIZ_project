import numpy as np
from PySide6.QtGui import QColor

_AF_BLUE = (0, 83, 214)      # pLDDT >= 90
_AF_CYAN = (101, 203, 243)   # pLDDT 70-89
_AF_YELLOW = (255, 219, 19)  # pLDDT 50-69
_AF_ORANGE = (255, 125, 69)  # pLDDT < 50
_AF_GRAY = (128, 128, 128)   # NaN

SS_COLORS: dict[int, QColor] = {
    0: QColor(230, 70, 90),    # H — red/magenta
    1: QColor(240, 200, 50),   # E — gold/yellow
    2: QColor(190, 190, 190),  # C — light gray
}
SS_LETTERS: dict[int, str] = {0: "H", 1: "E", 2: "C"}
SS_NAMES: dict[int, str] = {0: "Helix", 1: "Sheet", 2: "Coil"}


def ss_color(label: int) -> QColor:
    return SS_COLORS.get(int(label), SS_COLORS[2])

_DIV_NEG = np.array([220, 50, 47], dtype=np.float32)    # red (negative delta)
_DIV_MID = np.array([255, 255, 255], dtype=np.float32)  # white (no change)
_DIV_POS = np.array([38, 139, 210], dtype=np.float32)   # blue (positive delta)


def _pack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack float R/G/B arrays (0-255) into uint32 0xFF_RR_GG_BB (ARGB32)."""
    r8 = np.clip(r, 0, 255).astype(np.uint32)
    g8 = np.clip(g, 0, 255).astype(np.uint32)
    b8 = np.clip(b, 0, 255).astype(np.uint32)
    return (np.uint32(0xFF) << 24) | (r8 << 16) | (g8 << 8) | b8


def alphafold_color_array(plddt: np.ndarray) -> np.ndarray:
    """Map pLDDT values → packed ARGB32 uint32 (same shape as input)."""
    flat = np.asarray(plddt, dtype=np.float32).ravel()
    nan_mask = np.isnan(flat)

    r = np.select(
        [flat >= 90, flat >= 70, flat >= 50, ~nan_mask],
        [_AF_BLUE[0], _AF_CYAN[0], _AF_YELLOW[0], _AF_ORANGE[0]],
        default=_AF_GRAY[0],
    ).astype(np.float32)
    g = np.select(
        [flat >= 90, flat >= 70, flat >= 50, ~nan_mask],
        [_AF_BLUE[1], _AF_CYAN[1], _AF_YELLOW[1], _AF_ORANGE[1]],
        default=_AF_GRAY[1],
    ).astype(np.float32)
    b = np.select(
        [flat >= 90, flat >= 70, flat >= 50, ~nan_mask],
        [_AF_BLUE[2], _AF_CYAN[2], _AF_YELLOW[2], _AF_ORANGE[2]],
        default=_AF_GRAY[2],
    ).astype(np.float32)

    r[nan_mask] = _AF_GRAY[0]
    g[nan_mask] = _AF_GRAY[1]
    b[nan_mask] = _AF_GRAY[2]

    return _pack_rgb(r, g, b).reshape(plddt.shape)


def delta_color_array(delta: np.ndarray, vmax: float = 30.0) -> np.ndarray:
    """Map pLDDT delta values → packed ARGB32 uint32 (symmetric diverging red-white-blue)."""
    flat = np.asarray(delta, dtype=np.float32).ravel()
    nan_mask = np.isnan(flat)

    t = np.clip(flat / vmax, -1.0, 1.0)  # in [-1, 1]

    # Negative t: lerp red → white; positive t: lerp white → blue
    neg = t < 0
    pos = ~neg

    r = np.empty_like(flat)
    g = np.empty_like(flat)
    b = np.empty_like(flat)

    # Negative side: t in [-1,0], map to [0,1] fraction from neg to mid
    frac_neg = 1.0 + t[neg]   # 0 at t=-1, 1 at t=0
    r[neg] = _DIV_NEG[0] + frac_neg * (_DIV_MID[0] - _DIV_NEG[0])
    g[neg] = _DIV_NEG[1] + frac_neg * (_DIV_MID[1] - _DIV_NEG[1])
    b[neg] = _DIV_NEG[2] + frac_neg * (_DIV_MID[2] - _DIV_NEG[2])

    # Positive side: t in [0,1], lerp mid → pos
    frac_pos = t[pos]
    r[pos] = _DIV_MID[0] + frac_pos * (_DIV_POS[0] - _DIV_MID[0])
    g[pos] = _DIV_MID[1] + frac_pos * (_DIV_POS[1] - _DIV_MID[1])
    b[pos] = _DIV_MID[2] + frac_pos * (_DIV_POS[2] - _DIV_MID[2])

    r[nan_mask] = _AF_GRAY[0]
    g[nan_mask] = _AF_GRAY[1]
    b[nan_mask] = _AF_GRAY[2]

    return _pack_rgb(r, g, b).reshape(delta.shape)
