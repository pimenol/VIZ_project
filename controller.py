"""Shared selection state for all views. Views read and write only through this."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class SelectionController(QObject):
    currentStepChanged = Signal(int)
    residueHoveredChanged = Signal(int)   # -1 = none
    residueSelectedChanged = Signal(int)  # -1 = none
    comparisonStepsChanged = Signal(list)  # list[int]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        self._hovered: int = -1
        self._selected: int = -1
        self._comparison: list[int] = []

    # ------------------------------------------------------------------
    # Setters — emit only when the value actually changes
    # ------------------------------------------------------------------

    def setCurrentStep(self, step: int) -> None:
        if step != self._current_step:
            self._current_step = step
            self.currentStepChanged.emit(step)

    def setHoveredResidue(self, residue: int) -> None:
        if residue != self._hovered:
            self._hovered = residue
            self.residueHoveredChanged.emit(residue)

    def setSelectedResidue(self, residue: int) -> None:
        if residue != self._selected:
            self._selected = residue
            self.residueSelectedChanged.emit(residue)

    def setComparisonSteps(self, steps: list[int]) -> None:
        if steps != self._comparison:
            self._comparison = list(steps)
            self.comparisonStepsChanged.emit(list(steps))

    def toggleComparisonStep(self, step: int) -> None:
        steps = list(self._comparison)
        if step in steps:
            steps.remove(step)
        else:
            steps.append(step)
        self.setComparisonSteps(steps)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def hovered_residue(self) -> int:
        return self._hovered

    @property
    def selected_residue(self) -> int:
        return self._selected

    @property
    def comparison_steps(self) -> list[int]:
        return list(self._comparison)
