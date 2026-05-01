from PySide6.QtCore import QObject, Signal


class SelectionController(QObject):
    currentStepChanged = Signal(int)
    residueHoveredChanged = Signal(int)        # -1 = none
    residueSelectedChanged = Signal(int)       # -1 = none
    comparisonStepsChanged = Signal(list)      # list[int]
    ssClassFilterChanged = Signal(set)         # subset of {0, 1, 2}

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        self._hovered: int = -1
        self._selected: int = -1
        self._comparison: list[int] = []
        self._ss_class_filter: set[int] = {0, 1, 2}

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
        normalized = sorted(set(steps))
        if normalized != self._comparison:
            self._comparison = normalized
            self.comparisonStepsChanged.emit(list(normalized))

    def toggleComparisonStep(self, step: int) -> None:
        steps = set(self._comparison)
        steps.symmetric_difference_update({step})
        self.setComparisonSteps(list(steps))

    def setSsClassFilter(self, allowed: set[int]) -> None:
        normalized = {int(v) for v in allowed if 0 <= int(v) <= 2}
        if normalized != self._ss_class_filter:
            self._ss_class_filter = normalized
            self.ssClassFilterChanged.emit(set(normalized))

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
        return self._comparison

    @property
    def ss_class_filter(self) -> set[int]:
        return set(self._ss_class_filter)
