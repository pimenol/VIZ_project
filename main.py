import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from controller import SelectionController
from data import load_run, load_run_from_dir, PtttRun
from views.embedding_view import EmbeddingView
from views.heatmap import HeatmapView
from views.line_chart import LineChartView
from views.profile_view import ProfileView
from views.residue_detail import ResidueDetailDock
from views.ss_track import SecondaryStructureTrack

_SLIDER_WIDTH = 160
_SPIN_WIDTH = 55


class MainWindow(QMainWindow):
    def __init__(self, run: PtttRun) -> None:
        super().__init__()
        self.setWindowTitle("ProteinTTT Visualization")
        self.resize(1400, 900)
        self._run = run
        self._ctrl = SelectionController(self)

        self._build_toolbar()
        self._build_central(run)
        self._build_detail_dock(run)
        self._build_status_bar()
        self._build_shortcuts()
        self._wire_controller()

        default_cmp = sorted({0, run.best_step, run.n_steps - 1})
        self._ctrl.setComparisonSteps(default_cmp)
        self._ctrl.setCurrentStep(0)

    def _make_step_spinbox(self, max_value: int, width: int = _SPIN_WIDTH) -> QSpinBox:
        sb = QSpinBox()
        sb.setMinimum(0)
        sb.setMaximum(max_value)
        sb.setFixedWidth(width)
        return sb

    def _build_toolbar(self) -> None:
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        btn_load = QPushButton("Load…")
        btn_load.clicked.connect(self._on_load)
        tb.addWidget(btn_load)

        btn_demo = QPushButton("Demo")
        btn_demo.clicked.connect(self._on_demo)
        tb.addWidget(btn_demo)

        tb.addSeparator()

        tb.addWidget(QLabel(" Step:"))
        self._step_slider = QSlider(Qt.Horizontal)
        self._step_slider.setMinimum(0)
        self._step_slider.setMaximum(self._run.n_steps - 1)
        self._step_slider.setFixedWidth(_SLIDER_WIDTH)
        tb.addWidget(self._step_slider)

        self._step_spin = self._make_step_spinbox(self._run.n_steps - 1)
        tb.addWidget(self._step_spin)

        tb.addSeparator()

        tb.addWidget(QLabel(" Color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["AlphaFold", "Delta"])
        self._color_combo.currentTextChanged.connect(self._on_color_mode)
        tb.addWidget(self._color_combo)

        tb.addSeparator()

        tb.addWidget(QLabel(" Res:"))
        self._res_lo = self._make_step_spinbox(self._run.n_residues - 1)
        self._res_lo.setValue(0)
        tb.addWidget(self._res_lo)
        tb.addWidget(QLabel("–"))
        self._res_hi = self._make_step_spinbox(self._run.n_residues - 1)
        self._res_hi.setValue(self._run.n_residues - 1)
        tb.addWidget(self._res_hi)
        self._res_lo.valueChanged.connect(self._on_residue_range)
        self._res_hi.valueChanged.connect(self._on_residue_range)

        tb.addSeparator()

        self._ss_strat_check = QCheckBox("SS-stratified")
        self._ss_strat_check.setChecked(False)
        self._ss_strat_check.toggled.connect(self._on_ss_stratified_toggled)
        tb.addWidget(self._ss_strat_check)

        tb.addSeparator()

        tb.addWidget(QLabel(" SS:"))
        self._ss_filter_checks: dict[int, QCheckBox] = {}
        for label, name in ((0, "H"), (1, "E"), (2, "C")):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_ss_filter_toggled)
            tb.addWidget(cb)
            self._ss_filter_checks[label] = cb

        tb.addSeparator()

        btn_png = QPushButton("Save PNG…")
        btn_png.clicked.connect(self._on_save_png)
        tb.addWidget(btn_png)

        self._step_slider.valueChanged.connect(self._ctrl.setCurrentStep)
        self._step_spin.valueChanged.connect(self._ctrl.setCurrentStep)

    def _build_central(self, run: PtttRun) -> None:
        self._line_chart = LineChartView(run, self._ctrl, self)
        self._heatmap = HeatmapView(run, self._ctrl, self)
        self._profile = ProfileView(run, self._ctrl, self)
        self._embedding = EmbeddingView(run, self._ctrl, self)

        self._heatmap_ss_track = SecondaryStructureTrack(
            run, self._ctrl, plot_left=52.0, plot_width=600.0, parent=self,
        )
        heatmap_container = QWidget()
        h_layout = QVBoxLayout(heatmap_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)
        h_layout.addWidget(self._heatmap_ss_track)
        h_layout.addWidget(self._heatmap)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._line_chart)
        top_splitter.addWidget(self._embedding)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(heatmap_container)
        bottom_splitter.addWidget(self._profile)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        root_splitter = QSplitter(Qt.Vertical)
        root_splitter.addWidget(top_splitter)
        root_splitter.addWidget(bottom_splitter)
        root_splitter.setStretchFactor(0, 1)
        root_splitter.setStretchFactor(1, 1)

        self._top_splitter = top_splitter
        self._bottom_splitter = bottom_splitter
        self._root_splitter = root_splitter

        self.setCentralWidget(root_splitter)

    def _build_detail_dock(self, run: PtttRun) -> None:
        self._detail_dock = ResidueDetailDock(
            run, self._ctrl, self._embedding.coords_2d_data, self,
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self._detail_dock)
        self._detail_dock.hide()

    def _build_status_bar(self) -> None:
        self._status_label = QLabel()
        sb = QStatusBar(self)
        sb.addWidget(self._status_label, 1)
        self.setStatusBar(sb)
        self._update_status()

    def _build_shortcuts(self) -> None:
        def _sc(key: str, fn) -> None:
            QShortcut(QKeySequence(key), self).activated.connect(fn)

        _sc("Left", lambda: self._step_by(-1))
        _sc("Right", lambda: self._step_by(+1))
        _sc("Shift+Left", lambda: self._step_by(-10))
        _sc("Shift+Right", lambda: self._step_by(+10))
        _sc("Home", lambda: self._ctrl.setCurrentStep(0))
        _sc("End",lambda: self._ctrl.setCurrentStep(self._run.best_step))
        _sc("+", self._zoom_in)
        _sc("-", self._zoom_out)
        _sc("R", self._reset_zoom)
        _sc("F", self._fit_view)

    def _wire_controller(self) -> None:
        c = self._ctrl
        c.currentStepChanged.connect(self._on_current_step_changed)
        c.residueHoveredChanged.connect(self._update_status)
        c.residueSelectedChanged.connect(self._update_status)
        c.comparisonStepsChanged.connect(self._update_status)

    def _on_current_step_changed(self, step: int) -> None:
        self._step_slider.blockSignals(True)
        self._step_spin.blockSignals(True)
        self._step_slider.setValue(step)
        self._step_spin.setValue(step)
        self._step_slider.blockSignals(False)
        self._step_spin.blockSignals(False)
        self._update_status()

    def _update_status(self, *_) -> None:
        c = self._ctrl
        cmp_str = str(c.comparison_steps) if c.comparison_steps else "none"
        self._status_label.setText(
            f"Step: {c.current_step}  |  "
            f"Residue selected: {c.selected_residue if c.selected_residue >= 0 else '—'}  |  "
            f"Hovered: {c.hovered_residue if c.hovered_residue >= 0 else '—'}  |  "
            f"Comparison: {cmp_str}"
        )

    def _on_load(self) -> None:
        run_dir = QFileDialog.getExistingDirectory(self, "Select run directory")
        if not run_dir:
            return
        try:
            run = load_run_from_dir(Path(run_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._reload(run)

    def _on_demo(self) -> None:
        demo_dir = Path(__file__).parent / "data" / "logs" / "A0A7G9V4P1"
        self._reload(load_run_from_dir(demo_dir))

    def _reload(self, run: PtttRun) -> None:
        self._run = run
        self._step_slider.setMaximum(run.n_steps - 1)
        self._step_spin.setMaximum(run.n_steps - 1)
        self._res_lo.blockSignals(True)
        self._res_hi.blockSignals(True)
        self._res_lo.setMaximum(run.n_residues - 1)
        self._res_lo.setValue(0)
        self._res_hi.setMaximum(run.n_residues - 1)
        self._res_hi.setValue(run.n_residues - 1)
        self._res_lo.blockSignals(False)
        self._res_hi.blockSignals(False)
        self._line_chart.set_run(run)
        self._heatmap.set_run(run)
        self._profile.set_run(run)
        self._embedding.set_run(run)
        self._detail_dock.set_run(run)
        self._heatmap_ss_track.set_run(run)
        default_cmp = sorted({0, run.best_step, run.n_steps - 1})
        self._ctrl.setComparisonSteps(default_cmp)
        self._ctrl.setCurrentStep(0)
        self._ctrl.setSelectedResidue(-1)

    def _on_color_mode(self, mode: str) -> None:
        self._heatmap.set_color_mode("delta" if mode == "Delta" else "absolute")

    def _on_ss_stratified_toggled(self, checked: bool) -> None:
        self._line_chart.set_ss_visible(checked)

    def _on_ss_filter_toggled(self, _checked: bool) -> None:
        allowed = {label for label, cb in self._ss_filter_checks.items() if cb.isChecked()}
        self._ctrl.setSsClassFilter(allowed)

    def _on_residue_range(self) -> None:
        n = self._run.n_residues
        lo = max(0, min(self._res_lo.value(), n - 1))
        hi = max(0, min(self._res_hi.value(), n - 1))
        if lo > hi:
            return
        self._heatmap.set_residue_range(lo, hi)
        self._profile.set_residue_range(lo, hi)
        self._heatmap_ss_track.set_residue_range(lo, hi)
        self._embedding.set_residue_range(lo, hi)

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "view.png", "PNG (*.png)")
        if not path:
            return
        scene = self._heatmap.scene()
        sr = scene.sceneRect()
        img = QImage(int(sr.width()), int(sr.height()), QImage.Format_ARGB32)
        img.fill(Qt.white)
        p = QPainter(img)
        scene.render(p, QRectF(img.rect()), sr)
        p.end()
        img.save(path)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        for sp in (self._root_splitter, self._top_splitter, self._bottom_splitter):
            total = sum(sp.sizes())
            half = total // 2
            sp.setSizes([half, total - half])

    def _step_by(self, delta: int) -> None:
        s = max(0, min(self._run.n_steps - 1, self._ctrl.current_step + delta))
        self._ctrl.setCurrentStep(s)

    def _zoom_in(self) -> None:
        self._heatmap.scale(1.2, 1.2)

    def _zoom_out(self) -> None:
        self._heatmap.scale(1 / 1.2, 1 / 1.2)

    def _reset_zoom(self) -> None:
        self._heatmap.resetTransform()

    def _fit_view(self) -> None:
        self._heatmap.fitInView(self._heatmap.scene().sceneRect(), Qt.KeepAspectRatio)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ProteinTTT visualization")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--demo", action="store_true", help="Load A0A7G9V4P1 demo run")
    grp.add_argument("--tsv", type=Path, metavar="FILE", help="Metrics TSV path")
    p.add_argument("--pdbs", type=Path, metavar="DIR", help="PDB folder (required with --tsv)")
    p.add_argument(
        "--recompute-ss", action="store_true",
        help="Force secondary-structure recomputation, ignoring any cached ss_matrix.npy",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.demo and args.pdbs is None:
        print("error: --pdbs is required when using --tsv", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if args.demo:
        run = load_run_from_dir(Path(__file__).parent / "data" / "logs" / "A0A7G9V4P1")
    else:
        run = load_run(args.tsv, args.pdbs, recompute_ss=args.recompute_ss)

    win = MainWindow(run)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


