from __future__ import annotations

import logging
from typing import Callable

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import numpy

from aligner_gui.ui.trainer_widget import Ui_trainer_widget
from aligner_gui.shared import const, gui_util
from aligner_gui.shared.graph_widget import GraphWidget
from aligner_gui.shared.progress_general_dialog import ProgressGeneralDialog
from aligner_gui.viewmodels.trainer_viewmodel import TrainerViewModel
from aligner_engine.summary import TrainSummary, ResultSummary
from aligner_engine.const import PHASE_TYPE_TRAINING, PHASE_TYPE_VALIDATION

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class _GpuMemBar(QWidget):
    """Vertical GPU-memory gauge with proportional fill and 25/50/75 % tick marks.

    The current percentage is drawn inside the bar (bottom area) so it never
    overlaps with the external labels.  The 25/50/75 tick labels are drawn to
    the right of the bar column only; they do not overlap with the fill value.
    """

    _BAR_W    = 40                          # fill column width (px)
    _TICK_H   = 140                         # total widget height
    _VAL_H    = 20                          # reserved pixels at bottom for value text
    _TICK_LEN = 5                           # horizontal tick length (right side)
    _TICK_PAD = 3                           # gap between tick and label

    _COL_LOW    = QColor( 60, 170, 110)    # green  ≤ 70 %
    _COL_MID    = QColor(220, 155,  35)    # amber  70–90 %
    _COL_HIGH   = QColor(210,  60,  55)    # red    > 90 %
    _COL_BG     = QColor( 28,  32,  38)
    _COL_BORDER = QColor( 64,  74,  86)
    _COL_TICK   = QColor( 85, 100, 115)
    _COL_VAL    = QColor(220, 226, 232)    # value text colour

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        fm_tick  = QFontMetrics(self._tick_font())
        fm_val   = QFontMetrics(self._val_font())
        tick_label_w = fm_tick.boundingRect("100%").width() + 2
        val_label_w  = fm_val.boundingRect("100%").width() + 4
        bar_w = max(self._BAR_W, val_label_w)
        self.setFixedSize(bar_w + self._TICK_LEN + self._TICK_PAD + tick_label_w,
                          self._TICK_H)

    def setValue(self, pct: int) -> None:
        self._value = max(0, min(pct, 100))
        self.update()

    # ------------------------------------------------------------------

    @staticmethod
    def _tick_font() -> QFont:
        f = QFont()
        f.setPointSize(7)
        return f

    @staticmethod
    def _val_font() -> QFont:
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        return f

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W  = self._BAR_W
        H  = self._TICK_H
        VH = self._VAL_H   # bottom strip reserved for value text
        PAD = 2            # inner vertical padding above value strip

        bar_h = H - VH     # height of the gauge column (above value strip)

        # ── background (gauge column only) ─────────────────────────────
        p.setPen(QPen(self._COL_BORDER, 1))
        p.setBrush(self._COL_BG)
        p.drawRoundedRect(QRect(0, 0, W - 1, bar_h - 1), 4, 4)

        # ── coloured fill (bottom → top, within gauge column) ───────────
        drawable_h = bar_h - 2 * PAD
        fill_h = round(drawable_h * self._value / 100)
        if fill_h > 0:
            v = self._value
            color = (self._COL_HIGH if v > 90
                     else self._COL_MID if v > 70
                     else self._COL_LOW)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(
                QRect(2, bar_h - PAD - fill_h, W - 4, fill_h), 3, 3
            )

        # ── dashed guide lines + right-side ticks + labels ─────────────
        tick_font = self._tick_font()
        p.setFont(tick_font)
        fm_tick = QFontMetrics(tick_font)
        text_h  = fm_tick.ascent()

        dash_pen  = QPen(self._COL_TICK, 1, Qt.DotLine)
        solid_pen = QPen(self._COL_TICK, 1, Qt.SolidLine)

        for mark in (75, 50, 25):
            y = PAD + round(drawable_h * (1 - mark / 100))

            p.setPen(dash_pen)
            p.drawLine(1, y, W - 2, y)

            p.setPen(solid_pen)
            p.drawLine(W, y, W + self._TICK_LEN, y)

            p.setPen(QPen(self._COL_TICK, 1))
            p.drawText(W + self._TICK_LEN + self._TICK_PAD,
                       y + text_h // 2,
                       f"{mark}%")

        # ── current value — drawn BELOW the gauge column, never overlaps ─
        val_font = self._val_font()
        p.setFont(val_font)
        fm_val = QFontMetrics(val_font)
        val_text = f"{self._value}%"
        val_rect = QRect(0, bar_h, W, VH)
        p.setPen(QPen(self._COL_VAL, 1))
        p.drawText(val_rect, Qt.AlignHCenter | Qt.AlignVCenter, val_text)

        p.end()


class TrainerView(QWidget, Ui_trainer_widget):
    """View layer for the Trainer tab.

    Responsibilities
    ----------------
    * Build and initialise Qt widgets.
    * Forward user actions to the ViewModel via command methods (no business
      logic here).
    * React to ViewModel signals by updating widgets.
    * Show modal dialogs that are inherently UI concerns (ProgressGeneralDialog).

    The View holds **no** reference to the Model/session. All session access is
    delegated to :class:`~aligner_gui.viewmodels.trainer_viewmodel.TrainerViewModel`.
    """

    COLOR_TRAINING = """QProgressBar::chunk { background: red; }"""
    COLOR_VALIDATION = """QProgressBar::chunk { background: green; }"""

    def __init__(
        self,
        session,
        tester_reload_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.setupUi(self)

        # ViewModel — no view reference passed in
        self.viewmodel = TrainerViewModel(session, tester_reload_callback, parent=self)

        # ------------------------------------------------------------------
        # Extra widgets not in the .ui file
        # ------------------------------------------------------------------
        self._canvas_graph = GraphWidget(hide_legend=True)
        self._canvas_graph_metric = GraphWidget(hide_legend=True, limit_range=(0, 1))
        self.layout_visualization.addWidget(self._canvas_graph)
        self.layout_visualization.addWidget(self._canvas_graph_metric)
        self.spin_resize.lineEdit().setReadOnly(True)
        self.spin_batch_size.setMaximum(256)
        self.spin_batch_size.setSingleStep(4)
        self.btn_train.setText("Train")
        self.check_resume = QCheckBox("Resume Last")
        self.check_resume.setToolTip("Continue training from the latest saved checkpoint.")
        self.horizontalLayout_4.addWidget(self.check_resume)

        self._build_device_panel()
        self._init_model_profile_ui()

        self._busy_indicator = QtGui.QMovie(
            "aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif"
        )
        self.lbl_train_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_train_indicator.hide()

        # ------------------------------------------------------------------
        # Sync initial UI state from settings
        # ------------------------------------------------------------------
        settings = self.viewmodel.get_settings()
        self.check_hflip.setChecked(settings.aug_flip_horizontal_use)
        self.check_vflip.setChecked(settings.aug_flip_vertical_use)
        self.check_no_rotation.setChecked(settings.no_rotation)
        self.check_include_empty.setChecked(settings.include_empty)
        self.spin_resize.setValue(settings.resize)
        self.spin_batch_size.setValue(settings.batch_size)
        self.spin_max_epochs.setValue(settings.max_epochs)

        # ------------------------------------------------------------------
        # View → ViewModel: user input commands
        # ------------------------------------------------------------------
        self.btn_train.clicked.connect(self._clicked_btn_train)
        self.check_hflip.clicked.connect(
            lambda: self.viewmodel.update_setting(aug_flip_horizontal_use=self.check_hflip.isChecked())
        )
        self.check_vflip.clicked.connect(
            lambda: self.viewmodel.update_setting(aug_flip_vertical_use=self.check_vflip.isChecked())
        )
        self.check_no_rotation.clicked.connect(
            lambda: self.viewmodel.update_setting(no_rotation=self.check_no_rotation.isChecked())
        )
        self.check_include_empty.clicked.connect(
            lambda: self.viewmodel.update_setting(include_empty=self.check_include_empty.isChecked())
        )
        self.spin_resize.valueChanged.connect(lambda v: self.viewmodel.update_setting(resize=v))
        self.spin_batch_size.valueChanged.connect(lambda v: self.viewmodel.update_setting(batch_size=v))
        self.spin_max_epochs.valueChanged.connect(lambda v: self.viewmodel.update_setting(max_epochs=v))
        self.table_training_history.clicked.connect(self._clicked_table_training_history)

        # ------------------------------------------------------------------
        # ViewModel → View: state-update signals
        # ------------------------------------------------------------------
        self.viewmodel.training_started.connect(self._on_training_started)
        self.viewmodel.training_stopped.connect(self._on_training_stopped)
        self.viewmodel.epoch_updated.connect(self._on_epoch_updated)
        self.viewmodel.iter_updated.connect(self._on_iter_updated)
        self.viewmodel.training_time_updated.connect(self.label_time.setText)
        self.viewmodel.status_label_updated.connect(self._on_status_label_updated)
        self.viewmodel.resume_state_changed.connect(self._on_resume_state_changed)
        self.viewmodel.device_usage_updated.connect(self._on_device_usage_updated)

        self.viewmodel.refresh_resume_state()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _build_device_panel(self) -> None:
        self.device_panel = QFrame()
        self.device_panel.setMinimumWidth(120)
        self.device_panel.setMaximumWidth(160)
        self.device_panel.setFrameShape(QFrame.StyledPanel)
        self.device_panel.setFrameShadow(QFrame.Raised)
        layout = QVBoxLayout(self.device_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)

        self.label_device_title = QLabel("GPU Mem")
        self.label_device_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_device_title)

        self.progress_device_usage = _GpuMemBar()
        self.progress_device_usage.hide()
        layout.addWidget(self.progress_device_usage, alignment=Qt.AlignHCenter)

        self.label_runtime_info = QLabel("VRAM --/--")
        self.label_runtime_info.setAlignment(Qt.AlignCenter)
        self.label_runtime_info.setStyleSheet("color: rgb(170, 180, 190);")
        self.label_runtime_info.setWordWrap(False)
        layout.addWidget(self.label_runtime_info)
        layout.addStretch(1)

        self.layout_visualization.addWidget(self.device_panel)

    def _init_model_profile_ui(self) -> None:
        self.lbl_model_profile = QLabel("Model")
        self.lbl_model_profile.setMinimumSize(100, 0)
        self.combo_model_profile = QComboBox()
        self.combo_model_profile.setMinimumSize(180, 0)
        self.combo_model_profile.setToolTip(
            "Choose the detector profile to train and export.\n"
            "This is stored in the project config so different projects can use different model families."
        )
        self.horizontalLayout_4.insertWidget(0, self.combo_model_profile)
        self.horizontalLayout_4.insertWidget(0, self.lbl_model_profile)

        settings = self.viewmodel.get_settings()
        current_profile = settings.model_profile
        for profile in self.viewmodel.get_model_profiles():
            self.combo_model_profile.addItem(profile.label, profile.id)
            idx = self.combo_model_profile.count() - 1
            self.combo_model_profile.setItemData(idx, profile.description, Qt.ToolTipRole)

        selected_index = self.combo_model_profile.findData(current_profile)
        if selected_index < 0:
            selected_index = self.combo_model_profile.findData("rotated_rtmdet_s")
        if selected_index >= 0:
            self.combo_model_profile.setCurrentIndex(selected_index)
            selected_profile_id = self.combo_model_profile.itemData(selected_index)
            if settings.model_profile != selected_profile_id:
                self.viewmodel.update_setting(model_profile=selected_profile_id)

        self.combo_model_profile.setEnabled(False)
        self.combo_model_profile.setToolTip(
            "Model profile switching is temporarily disabled until the corresponding pretrained weights are bundled."
        )
        self.lbl_model_profile.hide()
        self.combo_model_profile.hide()

    # ------------------------------------------------------------------
    # Input handlers (View → ViewModel commands)
    # ------------------------------------------------------------------

    def _clicked_btn_train(self) -> None:
        if self.btn_train.isChecked():
            self._request_start_training()
        else:
            self.viewmodel.stop_training("The process has been terminated at the user's request.")

    def _request_start_training(self) -> None:
        resume = self.check_resume.isChecked()

        ok, error = self.viewmodel.validate_start_training(resume)
        if not ok:
            self.btn_train.setChecked(False)
            if error == "no_checkpoint":
                self.viewmodel.refresh_resume_state()
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "No last checkpoint was found.\nRun at least one epoch first or start a fresh training.",
                )
            elif error == "max_epochs_reached":
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "The latest checkpoint already reached the configured max epochs.\n"
                    "Increase Epochs to continue training.",
                )
            return

        # Tell ViewModel preparation is beginning (emits training_started → UI update)
        self.viewmodel.begin_training_prep(resume)

        # Run preparation dialog (UI-only concern: modal blocking dialog)
        dlg = ProgressGeneralDialog(
            "Preparing training...",
            self.viewmodel.prepare_training_assets,
            2,
        )
        dlg.exec_()

        if not dlg.is_success:
            gui_util.get_message_box(
                self,
                "Invalid Dataset",
                "Failed to prepare the dataset.\n"
                "Choose images in the labeler first and make sure enough labeled data exists.",
            )
            self.viewmodel.abort_training_prep()
            return

        self.viewmodel.launch_training(resume)

    def _clicked_table_training_history(self) -> None:
        epoch = self.table_training_history.currentRow() + 1
        self._refresh_validation_table(epoch, self.viewmodel.get_valid_result_summary())

    # ------------------------------------------------------------------
    # ViewModel signal slots (update UI in response to state changes)
    # ------------------------------------------------------------------

    def _set_training_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all training configuration widgets at once."""
        self.check_hflip.setEnabled(enabled)
        self.check_vflip.setEnabled(enabled)
        self.check_no_rotation.setEnabled(enabled)
        self.check_include_empty.setEnabled(enabled)
        self.spin_resize.setEnabled(enabled)
        self.spin_batch_size.setEnabled(enabled)
        self.spin_max_epochs.setEnabled(enabled)
        # check_resume is controlled by resume_state_changed signal
        # combo_model_profile stays disabled (feature not yet enabled)

    def _on_training_started(self, is_resume: bool, start_epoch: int) -> None:
        settings = self.viewmodel.get_settings()

        self.btn_train.setChecked(True)
        self.btn_train.setText("Stop")
        self.btn_train.setEnabled(True)

        if is_resume and start_epoch > 0:
            self.label_time.setText(f"Resuming from epoch {start_epoch + 1}...")
        else:
            self.label_time.setText("Preparing training...")

        self.label_device_title.setText("GPU Mem")
        self.label_runtime_info.setText("VRAM --/--")

        if is_resume and start_epoch > 0:
            self._refresh_existing_training_monitor(start_epoch)
        else:
            self._init_training_chart()
            self._init_training_history()
            self._init_validation_table()

        self.progress_epoch.setMaximum(settings.max_epochs)
        self.progress_epoch.setValue(start_epoch)
        self.progress_epoch.setStyleSheet(gui_util.get_dark_style())

        self.progress_iter.setMaximum(1)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.lbl_train_indicator.show()

        self._set_training_controls_enabled(False)
        self.check_resume.setEnabled(False)

        self.progress_device_usage.setValue(0)
        self.progress_device_usage.show()

    def _on_training_stopped(self, reason: str) -> None:
        self.btn_train.setChecked(False)
        self.btn_train.setText("Train")
        self.btn_train.setEnabled(True)

        self.progress_epoch.setMaximum(1)
        self.progress_epoch.setValue(0)
        self.progress_epoch.setStyleSheet(gui_util.get_dark_style())

        self.progress_iter.setMaximum(1)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.lbl_train_indicator.hide()

        self._set_training_controls_enabled(True)

        self.progress_device_usage.hide()
        self.progress_device_usage.setValue(0)
        self.label_device_title.setText("GPU Mem")
        self.label_runtime_info.setText("VRAM --/--")

    def _on_epoch_updated(self, epoch: int, _ckpt_path: str) -> None:
        self.progress_epoch.setValue(epoch)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self._refresh_training_chart(
            self.viewmodel.get_train_summary(),
            self.viewmodel.get_train_result_summary(),
            self.viewmodel.get_valid_result_summary(),
        )
        self._refresh_training_history_table(
            self.viewmodel.get_train_summary(),
            self.viewmodel.get_train_result_summary(),
            self.viewmodel.get_valid_result_summary(),
        )
        self._refresh_validation_table(epoch, self.viewmodel.get_valid_result_summary())

    def _on_iter_updated(self, phase: str, idx: int, total: int) -> None:
        if phase == PHASE_TYPE_TRAINING:
            self.progress_iter.setStyleSheet(self.COLOR_TRAINING)
        elif phase == PHASE_TYPE_VALIDATION:
            self.progress_iter.setStyleSheet(self.COLOR_VALIDATION)
        self.progress_iter.setMaximum(total)
        self.progress_iter.setValue(idx + 1)

    def _on_status_label_updated(self, message: str) -> None:
        # Only show thread status messages before any real timer data is available.
        if self.progress_epoch.value() == 0 and self.progress_iter.value() == 0:
            self.label_time.setText(message)

    def _on_resume_state_changed(self, can_resume: bool, last_epoch: int) -> None:
        self.check_resume.setEnabled(can_resume)
        if not can_resume:
            self.check_resume.setChecked(False)
            self.check_resume.setToolTip("Enable after at least one checkpoint has been saved.")
            return
        resume_epoch = last_epoch + 1 if last_epoch > 0 else 1
        self.check_resume.setToolTip(
            f"Continue training from auto_saved\\last.pth.\nNext epoch: {resume_epoch}"
        )

    def _on_device_usage_updated(self, info: dict) -> None:
        if not info.get("visible", False):
            self.progress_device_usage.hide()
            self.label_device_title.setText(info.get("title", "CPU"))
            self.label_runtime_info.setText(info.get("info", ""))
            return
        self.progress_device_usage.show()
        self.progress_device_usage.setValue(info.get("value", 0))
        self.label_device_title.setText(info.get("title", "GPU Mem"))
        self.label_runtime_info.setText(info.get("info", ""))
        self.label_runtime_info.setToolTip(info.get("tooltip", ""))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def show(self) -> None:
        super().show()
        self.viewmodel.refresh_resume_state()
        train_summary = self.viewmodel.get_train_summary()
        if not train_summary.tr_by_epoch:
            return
        try:
            last_epoch = list(train_summary.tr_by_epoch.keys())[-1]
            train_result = self.viewmodel.get_train_result_summary()
            valid_result = self.viewmodel.get_valid_result_summary()
            self._refresh_training_chart(train_summary, train_result, valid_result)
            self._refresh_training_history_table(train_summary, train_result, valid_result)
            self._refresh_validation_table(last_epoch, valid_result)
        except Exception:
            pass

    def close(self) -> bool:
        self.viewmodel.close()
        return super().close()

    # ------------------------------------------------------------------
    # Chart / table helpers (pure rendering — no business logic)
    # ------------------------------------------------------------------

    def _refresh_existing_training_monitor(self, last_epoch: int) -> None:
        train_summary = self.viewmodel.get_train_summary()
        if not train_summary.tr_by_epoch:
            return
        try:
            train_result = self.viewmodel.get_train_result_summary()
            valid_result = self.viewmodel.get_valid_result_summary()
            self._refresh_training_chart(train_summary, train_result, valid_result)
            self._refresh_training_history_table(train_summary, train_result, valid_result)
            self._refresh_validation_table(last_epoch, valid_result)
        except Exception:
            pass

    def _init_training_chart(self) -> None:
        metric_name = self.viewmodel.metric_name
        self._canvas_graph.reset()
        self._canvas_graph_metric.reset()
        self._canvas_graph.setName("Train Loss")
        self._canvas_graph.setBottomName("Epoch")
        self._canvas_graph.setLine("Loss", [], QtGui.QColor(205, 92, 92), 2)
        self._canvas_graph_metric.setName(f"Validation {metric_name}")
        self._canvas_graph_metric.setBottomName("Epoch")
        self._canvas_graph_metric.setLine(metric_name, [], QtGui.QColor(92, 205, 92), 2)

    def _init_training_history(self) -> None:
        self.table_training_history.clear()
        h_names = [
            "Train\nLoss", f"Valid\n{self.viewmodel.metric_name}",
            "Corner\nError", "Corner\nX", "Corner\nY",
            "Center\nError", "Center\nX", "Center\nY",
            "Longside\nError", "Shortside\nError", "Update",
        ]
        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(1)

    def _init_validation_table(self) -> None:
        self.table_validation.clear()

    def _refresh_training_chart(
        self,
        train_summary: TrainSummary,
        train_result_summary: ResultSummary,
        valid_result_summary: ResultSummary,
    ) -> None:
        metric_name = self.viewmodel.metric_name
        tr_loss, va_metric = [], []
        for epoch in train_summary.tr_by_epoch:
            tr_loss.append(train_summary.tr_by_epoch[epoch]["loss"])
            if epoch in train_summary.va_by_epoch:
                if epoch in valid_result_summary.map:
                    va_metric.append(valid_result_summary.get_metric(epoch, metric_name))
                else:
                    va_metric.append(va_metric[-1] if va_metric else 0)
            else:
                va_metric.append(numpy.nan)

        self._canvas_graph.setName("Train Loss")
        self._canvas_graph.setLine("Loss", tr_loss, QtGui.QColor(205, 92, 92), 2)
        self._canvas_graph_metric.setName(f"Validation {metric_name}")
        self._canvas_graph_metric.setLine(metric_name, va_metric, QtGui.QColor(92, 205, 92), 2)

    def _refresh_training_history_table(
        self,
        train_summary: TrainSummary,
        train_result_summary: ResultSummary,
        valid_result_summary: ResultSummary,
    ) -> None:
        metric_name = self.viewmodel.metric_name
        self.table_training_history.clear()

        h_names = [
            "Train\nLoss", f"Valid\n{metric_name}",
            "Corner\nError", "Corner\nX", "Corner\nY",
            "Center\nError", "Center\nX", "Center\nY",
            "Longside\nError", "Shortside\nError", "Update",
        ]
        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(len(train_summary.tr_by_epoch))

        header_epoch_map: dict[int, int] = {}
        for idx, (ep, info) in enumerate(train_summary.tr_by_epoch.items()):
            self.table_training_history.setVerticalHeaderItem(idx, QTableWidgetItem(str(ep)))
            self.table_training_history.setItem(idx, 0, QTableWidgetItem("%.4lf" % info["loss"]))
            header_epoch_map[ep] = idx

        for ep, _info in train_summary.va_by_epoch.items():
            idx = header_epoch_map[ep]
            self.table_training_history.setItem(
                idx, 1,
                QTableWidgetItem("%.4lf" % valid_result_summary.get_metric(ep, metric_name)),
            )
            mpe = valid_result_summary.get_metric(ep, "mPE")
            for col, key in enumerate(
                ["corner_error", "corner_dx", "corner_dy",
                 "center_error", "center_dx", "center_dy",
                 "longside", "shortside"],
                start=2,
            ):
                self.table_training_history.setItem(
                    idx, col, QTableWidgetItem(f"{mpe[key]:.1f}")
                )

        for updated_epoch in train_summary.update_model_epoch:
            idx = header_epoch_map[updated_epoch]
            self.table_training_history.setItem(idx, 10, QTableWidgetItem("O"))

        self.table_training_history.resizeColumnsToContents()
        self.table_training_history.resizeRowsToContents()
        self.table_training_history.scrollToBottom()

    def _refresh_validation_table(
        self, epoch: int, valid_result_summary: ResultSummary
    ) -> None:
        self.table_validation.clear()
        self.table_validation.clearSpans()
        n_class = len(valid_result_summary.class_name)
        if n_class <= 0:
            return

        metric_name = self.viewmodel.metric_name
        h_class_names = [valid_result_summary.class_name[i] for i in range(n_class)]
        self.table_validation.setColumnCount(len(h_class_names))
        self.table_validation.setHorizontalHeaderLabels(h_class_names)
        v_row_names = [
            "#", "AP", "Epoch", metric_name,
            "Corner Error", "Corner X", "Corner Y",
            "Center Error", "Center X", "Center Y",
            "Width Error", "Height Error",
        ]
        self.table_validation.setRowCount(len(v_row_names))
        self.table_validation.setVerticalHeaderLabels(v_row_names)

        def format_metric(value):
            if value is None:
                return "-"
            try:
                if numpy.isnan(value):
                    return "-"
            except TypeError:
                pass
            return f"{value:.1f}"

        if epoch in valid_result_summary.aps:
            aps = valid_result_summary.aps[epoch]
            if aps is not None:
                for c in range(n_class):
                    for row, val in ((0, str(int(aps[c][1]))), (1, "%.2lf" % aps[c][0])):
                        item = QTableWidgetItem(val)
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setBackground(QtGui.QColor(144, 164, 174))
                        self.table_validation.setItem(row, c, item)

        def set_overall_row(row_no: int, text: str) -> None:
            self.table_validation.setSpan(row_no, 0, 1, n_class)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QtGui.QColor(65, 78, 92))
            self.table_validation.setItem(row_no, 0, item)

        set_overall_row(2, str(epoch))

        if epoch in valid_result_summary.map:
            set_overall_row(3, "%.4lf" % valid_result_summary.map[epoch])

        if epoch in valid_result_summary.mpe:
            mpe = valid_result_summary.get_metric(epoch, "mPE")
            class_metrics = valid_result_summary.mpe_by_class.get(epoch, {})
            row_key_map = {
                4: "corner_error", 5: "corner_dx", 6: "corner_dy",
                7: "center_error", 8: "center_dx", 9: "center_dy",
                10: "longside", 11: "shortside",
            }
            if class_metrics:
                for row_no, metric_key in row_key_map.items():
                    for c in range(n_class):
                        value = class_metrics.get(c, {}).get(metric_key)
                        item = QTableWidgetItem(format_metric(value))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table_validation.setItem(row_no, c, item)
            else:
                for row_no, metric_key in row_key_map.items():
                    set_overall_row(row_no, f"{mpe[metric_key]:.1f}")

        self.table_validation.resizeColumnsToContents()
        self.table_validation.resizeRowsToContents()
