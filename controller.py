from PySide6.QtCore import QObject, Signal


class SelectionController(QObject):
    currentStepChanged = Signal(int)
    residueHoveredChanged = Signal(int)        # -1 = none
    residueSelectedChanged = Signal(int)       # -1 = none
    comparisonStepsChanged = Signal(list)      # list[int]
    stepRangeSelected = Signal(int, int)       # (-1, -1) = clear
    residueRangeSelected = Signal(int, int)    # (-1, -1) = clear
    comparisonResiduesChanged = Signal(list)   # list[int]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_step: int = 0
        self._hovered: int = -1
        self._selected: int = -1
        self._comparison: list[int] = []
        self._step_range: tuple[int, int] = (-1, -1)
        self._residue_range: tuple[int, int] = (-1, -1)
        self._comparison_residues: list[int] = []

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

    def setStepRange(self, lo: int, hi: int) -> None:
        rng = (lo, hi) if lo <= hi else (hi, lo)
        if (lo, hi) == (-1, -1):
            rng = (-1, -1)
        if rng != self._step_range:
            self._step_range = rng
            self.stepRangeSelected.emit(rng[0], rng[1])

    def setResidueRange(self, lo: int, hi: int) -> None:
        rng = (lo, hi) if lo <= hi else (hi, lo)
        if (lo, hi) == (-1, -1):
            rng = (-1, -1)
        if rng != self._residue_range:
            self._residue_range = rng
            self.residueRangeSelected.emit(rng[0], rng[1])

    def setComparisonResidues(self, residues: list[int]) -> None:
        normalized = sorted(set(residues))
        if normalized != self._comparison_residues:
            self._comparison_residues = normalized
            self.comparisonResiduesChanged.emit(list(normalized))

    def toggleComparisonResidue(self, residue: int) -> None:
        residues = set(self._comparison_residues)
        residues.symmetric_difference_update({residue})
        self.setComparisonResidues(list(residues))

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
    def step_range(self) -> tuple[int, int]:
        return self._step_range

    @property
    def residue_range(self) -> tuple[int, int]:
        return self._residue_range

    @property
    def comparison_residues(self) -> list[int]:
        return self._comparison_residues
