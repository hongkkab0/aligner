from __future__ import annotations

import time
import traceback
from typing import Callable, Optional

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QProgressBar,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import logging
import numpy

from aligner_gui.ui.trainer_widget import Ui_trainer_widget
from aligner_gui.trainer.training_timer import TrainingTimer, timestamp2time
from aligner_gui.trainer.gpu_monitor import GpuUsageSnapshot, NvidiaSmiPoller
from aligner_gui.utils import const, gui_util
from aligner_gui.widgets.graph_widget import GraphWidget
from aligner_gui.widgets.progress_general_dialog import ProgressGeneralDialog
from aligner_gui.viewmodels.trainer_viewmodel import TrainerViewModel
from aligner_engine.summary import TrainSummary, ResultSummary
from aligner_engine.const import PHASE_TYPE_TRAINING, PHASE_TYPE_VALIDATION

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class TrainerWidget(QWidget, Ui_trainer_widget):
    """View layer for the Trainer tab.

    All business logic lives in :class:`~aligner_gui.viewmodels.trainer_viewmodel.TrainerViewModel`.
    This class is responsible only for:

    * Building and wiring up the Qt UI elements.
    * Connecting ViewModel signals to local UI-update slots.
    * Delegating user actions to the ViewModel via command methods.
    * Showing modal dialogs that cannot live in the ViewModel (ProgressGeneralDialog).
    """

    COLOR_TRAINING = """QProgressBar::chunk { background: red; }"""
    COLOR_VALIDATION = """QProgressBar::chunk { background: green; }"""
    DEVICE_BAR_STYLE = """
    QProgressBar {
        background-color: rgb(28, 32, 38);
        border: 1px solid rgb(64, 74, 86);
        border-radius: 6px;
        padding: 1px;
    }
    QProgressBar::chunk {
        background-color: rgb(60, 170, 110);
        border-radius: 4px;
        margin: 1px;
    }
    """

    def __init__(
        self,
        session,
        app_viewmodel,
        tester_reload_callback: Optional[Callable[[], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self._vm = TrainerViewModel(session, app_viewmodel, tester_reload_callback, parent=self)

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

        # -- Bind UI input handlers
        self.btn_train.clicked.connect(self._clicked_btn_train)
        self.check_hflip.clicked.connect(lambda: self._vm.update_setting(aug_flip_horizontal_use=self.check_hflip.isChecked()))
        self.check_vflip.clicked.connect(lambda: self._vm.update_setting(aug_flip_vertical_use=self.check_vflip.isChecked()))
        self.check_no_rotation.clicked.connect(lambda: self._vm.update_setting(no_rotation=self.check_no_rotation.isChecked()))
        self.check_include_empty.clicked.connect(lambda: self._vm.update_setting(include_empty=self.check_include_empty.isChecked()))
        self.spin_resize.valueChanged.connect(lambda v: self._vm.update_setting(resize=v))
        self.spin_batch_size.valueChanged.connect(lambda v: self._vm.update_setting(batch_size=v))
        self.spin_max_epochs.valueChanged.connect(lambda v: self._vm.update_setting(max_epochs=v))
        self.table_training_history.clicked.connect(self._clicked_table_training_history)

        # -- Sync initial UI values from settings
        settings = self._vm.get_settings()
        self.check_hflip.setChecked(settings.aug_flip_horizontal_use)
        self.check_vflip.setChecked(settings.aug_flip_vertical_use)
        self.check_no_rotation.setChecked(settings.no_rotation)
        self.check_include_empty.setChecked(settings.include_empty)
        self.spin_resize.setValue(settings.resize)
        self.spin_batch_size.setValue(settings.batch_size)
        self.spin_max_epochs.setValue(settings.max_epochs)

        # -- Connect ViewModel signals → View slots
        self._vm.training_prep_started.connect(self._on_training_prep_started)
        self._vm.training_stopped.connect(self._on_training_stopped)
        self._vm.epoch_updated.connect(self._on_epoch_updated)
        self._vm.iter_updated.connect(self._on_iter_updated)
        self._vm.status_message_changed.connect(self._on_status_message_changed)
        self._vm.resume_state_changed.connect(self._on_resume_state_changed)

        # -- Busy indicator
        self._busy_indicator = QtGui.QMovie("aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif")
        self.lbl_train_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_train_indicator.hide()

        # -- Training support objects
        self._training_timer = TrainingTimer()
        self._device_poll_timer = QTimer(self)
        self._device_poll_timer.setInterval(1500)
        self._device_poll_timer.timeout.connect(self._refresh_device_usage)
        self._gpu_monitor = NvidiaSmiPoller(self)
        self._gpu_monitor.stats_updated.connect(self._apply_gpu_snapshot)
        self._gpu_monitor.unavailable.connect(self._handle_gpu_monitor_unavailable)
        self._gpu_monitor_available: bool | None = None
        self._current_epoch_for_eta = 1
        self._last_iter_ui_update_ts = 0.0

        self._vm.refresh_resume_state()

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

        self.progress_device_usage = QProgressBar()
        self.progress_device_usage.setOrientation(Qt.Vertical)
        self.progress_device_usage.setFixedSize(20, 118)
        self.progress_device_usage.setRange(0, 100)
        self.progress_device_usage.setTextVisible(False)
        self.progress_device_usage.setInvertedAppearance(False)
        self.progress_device_usage.setStyleSheet(self.DEVICE_BAR_STYLE)
        self.progress_device_usage.hide()
        layout.addWidget(self.progress_device_usage, alignment=Qt.AlignHCenter)

        self.label_device_percent = QLabel("")
        self.label_device_percent.setAlignment(Qt.AlignCenter)
        self.label_device_percent.setStyleSheet("font-weight: 600; color: rgb(220, 226, 232);")
        layout.addWidget(self.label_device_percent)

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

        settings = self._vm.get_settings()
        current_profile = settings.model_profile
        for profile in self._vm.get_model_profiles():
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
                self._vm.update_setting(model_profile=selected_profile_id)

        self.combo_model_profile.setEnabled(False)
        self.combo_model_profile.setToolTip(
            "Model profile switching is temporarily disabled until the corresponding pretrained weights are bundled."
        )
        self.lbl_model_profile.hide()
        self.combo_model_profile.hide()

    # ------------------------------------------------------------------
    # Button handlers (View → ViewModel commands)
    # ------------------------------------------------------------------

    def _clicked_btn_train(self) -> None:
        if self.btn_train.isChecked():
            self._request_start_training()
        else:
            self._vm.stop_training("The process has been terminated at the user's request.")

    def _request_start_training(self) -> None:
        resume = self.check_resume.isChecked()

        ok, error = self._vm.validate_start_training(resume)
        if not ok:
            self.btn_train.setChecked(False)
            if error == "no_checkpoint":
                self._vm.refresh_resume_state()
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "No last checkpoint was found.\nRun at least one epoch first or start a fresh training.",
                )
            elif error == "max_epochs_reached":
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "The latest checkpoint already reached the configured max epochs.\nIncrease Epochs to continue training.",
                )
            return

        # Signal ViewModel to begin prep → triggers _on_training_prep_started
        self._vm.begin_training_prep(resume)

        # Show prep dialog (View concern: modal dialog)
        def work(processing_signal):
            return self._vm.prepare_training_assets(processing_signal)

        dlg = ProgressGeneralDialog("Preparing training...", work, 2)
        dlg.exec_()

        if not dlg.is_success:
            gui_util.get_message_box(
                self,
                "Invalid Dataset",
                "Failed to prepare the dataset.\nChoose images in the labeler first and make sure enough labeled data exists.",
            )
            self._vm.abort_training_prep()
            return

        torch = self._get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._vm.launch_training(resume)

    def _clicked_table_training_history(self) -> None:
        epoch = self.table_training_history.currentRow() + 1
        self._refresh_validation_table(epoch, self._vm.get_valid_result_summary())

    # ------------------------------------------------------------------
    # ViewModel signal slots (update UI in response to state changes)
    # ------------------------------------------------------------------

    def _on_training_prep_started(self, is_resume: bool, start_epoch: int) -> None:
        """Called when the ViewModel confirms training preparation has begun."""
        settings = self._vm.get_settings()

        self.btn_train.setChecked(True)
        self.btn_train.setText("Stop")
        self.btn_train.setEnabled(True)

        if is_resume and start_epoch > 0:
            self.label_time.setText(f"Resuming from epoch {start_epoch + 1}...")
        else:
            self.label_time.setText("Preparing training...")

        self.label_device_title.setText("GPU Mem")
        self.label_device_percent.setText("")
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

        self.check_hflip.setEnabled(False)
        self.check_vflip.setEnabled(False)
        self.check_no_rotation.setEnabled(False)
        self.check_include_empty.setEnabled(False)
        self.check_resume.setEnabled(False)
        self.combo_model_profile.setEnabled(False)
        self.spin_resize.setEnabled(False)
        self.spin_batch_size.setEnabled(False)
        self.spin_max_epochs.setEnabled(False)

        self._training_timer.train_start(start_epoch, settings.max_epochs)
        self._current_epoch_for_eta = max(start_epoch + 1, 1)
        self._last_iter_ui_update_ts = 0.0
        self._gpu_monitor_available = None
        self.progress_device_usage.setValue(0)
        self.progress_device_usage.show()

        torch = self._get_torch()
        if torch.cuda.is_available():
            self._gpu_monitor.set_preferred_device_index(torch.cuda.current_device())
        self._device_poll_timer.start()
        self._refresh_device_usage()

    def _on_training_stopped(self, reason: str) -> None:
        """Called when the ViewModel reports training has ended."""
        torch = self._get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reason == const.SUCCESS:
            TRAIN_LOGGER.info("training finished successfully")
        elif reason == const.ERROR:
            TRAIN_LOGGER.error("training failed")
        else:
            TRAIN_LOGGER.info(reason)

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

        self.check_hflip.setEnabled(True)
        self.check_vflip.setEnabled(True)
        self.check_no_rotation.setEnabled(True)
        self.check_include_empty.setEnabled(True)
        self.combo_model_profile.setEnabled(False)
        self.spin_resize.setEnabled(True)
        self.spin_batch_size.setEnabled(True)
        self.spin_max_epochs.setEnabled(True)

        self._device_poll_timer.stop()
        self._gpu_monitor.stop()
        self._gpu_monitor_available = None
        self.progress_device_usage.hide()
        self.progress_device_usage.setValue(0)
        self.label_device_percent.setText("")
        self.label_device_title.setText("GPU Mem")
        self.label_runtime_info.setText("VRAM --/--")

        self._vm.refresh_resume_state()

    def _on_epoch_updated(self, current_epoch: int, current_ckpt_path: str) -> None:
        self.progress_epoch.setValue(current_epoch)
        self._current_epoch_for_eta = current_epoch + 1
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())

        self._vm.save_records_after_epoch(current_epoch, current_ckpt_path)
        self._refresh_training_time(current_epoch)
        self._refresh_training_chart(
            self._vm.get_train_summary(),
            self._vm.get_train_result_summary(),
            self._vm.get_valid_result_summary(),
        )
        self._refresh_training_history_table(
            self._vm.get_train_summary(),
            self._vm.get_train_result_summary(),
            self._vm.get_valid_result_summary(),
        )
        self._refresh_validation_table(current_epoch, self._vm.get_valid_result_summary())

    def _on_iter_updated(self, phase_type: str, iter_idx: int, iter_len: int) -> None:
        if phase_type == PHASE_TYPE_TRAINING:
            self.progress_iter.setStyleSheet(self.COLOR_TRAINING)
        elif phase_type == PHASE_TYPE_VALIDATION:
            self.progress_iter.setStyleSheet(self.COLOR_VALIDATION)
        else:
            raise NotImplementedError(phase_type)

        self.progress_iter.setMaximum(iter_len)
        self.progress_iter.setValue(iter_idx + 1)

        now = time.monotonic()
        is_last_iter = iter_len > 0 and (iter_idx + 1) >= iter_len
        if is_last_iter or (now - self._last_iter_ui_update_ts) >= 0.15:
            self._last_iter_ui_update_ts = now
            self._refresh_training_time_live(phase_type, iter_idx, iter_len)

    def _on_status_message_changed(self, message: str) -> None:
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

    # ------------------------------------------------------------------
    # GPU / device monitoring (View-only concern)
    # ------------------------------------------------------------------

    def _get_torch(self):
        import torch
        return torch

    def _refresh_device_usage(self) -> None:
        torch = self._get_torch()
        if not torch.cuda.is_available():
            self.progress_device_usage.hide()
            self.label_device_title.setText("CPU")
            self.label_device_percent.setText("")
            self.label_runtime_info.setText("Device: CPU training")
            return

        if self._gpu_monitor_available is not True:
            self._apply_gpu_memory_fallback()
        self._gpu_monitor.poll()

    def _apply_gpu_snapshot(self, snapshot: GpuUsageSnapshot) -> None:
        self._gpu_monitor_available = True
        memory_percent = max(0.0, min(snapshot.memory_percent, 100.0))
        used_gb = snapshot.memory_used_mb / 1024.0
        total_gb = snapshot.memory_total_mb / 1024.0
        self.progress_device_usage.show()
        self.progress_device_usage.setValue(int(round(memory_percent)))
        self.label_device_title.setText("GPU Mem")
        self.label_device_percent.setText(f"{memory_percent:.0f}%")
        self.label_runtime_info.setText(f"{used_gb:.1f}/{total_gb:.1f} GB ({memory_percent:.0f}%)")
        self.label_runtime_info.setToolTip(snapshot.name)

    def _handle_gpu_monitor_unavailable(self, _: str) -> None:
        self._gpu_monitor_available = False
        self._apply_gpu_memory_fallback()

    def _apply_gpu_memory_fallback(self) -> None:
        gpu_stats = self._sample_gpu_memory_stats()
        if gpu_stats is None:
            self.progress_device_usage.hide()
            self.label_device_title.setText("CPU")
            self.label_device_percent.setText("")
            self.label_runtime_info.setText("Device: CPU training")
            return

        self.progress_device_usage.show()
        self.progress_device_usage.setValue(int(round(max(0, min(gpu_stats["memory_percent"], 100)))))
        self.label_device_title.setText("GPU Mem")
        self.label_device_percent.setText(f'{gpu_stats["memory_percent"]:.0f}%')
        self.label_runtime_info.setText(gpu_stats["text"])
        self.label_runtime_info.setToolTip(gpu_stats["device_name"])

    def _sample_gpu_memory_stats(self) -> dict | None:
        torch = self._get_torch()
        if not torch.cuda.is_available():
            return None

        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(device_index)
            total_memory = max(total_memory, 1)
            used_memory = max(total_memory - free_memory, 0)
        except Exception:
            props = torch.cuda.get_device_properties(device_index)
            total_memory = max(props.total_memory, 1)
            allocated_memory = torch.cuda.memory_allocated(device_index)
            reserved_memory = torch.cuda.memory_reserved(device_index)
            used_memory = max(allocated_memory, reserved_memory)
        usage = used_memory * 100.0 / total_memory
        return {
            "device_name": device_name,
            "memory_percent": usage,
            "text": f"{used_memory / (1024 ** 3):.1f}/{total_memory / (1024 ** 3):.1f} GB ({usage:.0f}%)",
        }

    # ------------------------------------------------------------------
    # Chart / table refreshes (View-only, uses data from ViewModel)
    # ------------------------------------------------------------------

    def _refresh_training_time(self, cur_epoch: int) -> None:
        one_epoch, avg_one_epoch, processed, remaining = self._training_timer.one_epoch_done(cur_epoch)
        self.label_time.setText(
            "Avg epoch %.1fs   Elapsed %s   Remaining %s"
            % (avg_one_epoch, timestamp2time(processed), timestamp2time(remaining))
        )

    def _refresh_training_time_live(self, phase_type: str, iter_idx: int, iter_len: int) -> None:
        current_epoch = min(max(self.progress_epoch.value() + 1, 1), max(self.progress_epoch.maximum(), 1))
        self._current_epoch_for_eta = current_epoch
        avg_step, processed, remaining = self._training_timer.one_iter_progress(
            phase_type, iter_idx, iter_len, current_epoch
        )
        phase_name = "Train" if phase_type == PHASE_TYPE_TRAINING else "Valid"
        self.label_time.setText(
            f"{phase_name} {iter_idx + 1}/{iter_len}   Avg step {avg_step:.2f}s"
            f"   Elapsed {timestamp2time(processed)}   Remaining {timestamp2time(remaining)}"
        )

    def _refresh_existing_training_monitor(self, last_epoch: int) -> None:
        self._refresh_training_chart(
            self._vm.get_train_summary(),
            self._vm.get_train_result_summary(),
            self._vm.get_valid_result_summary(),
        )
        self._refresh_training_history_table(
            self._vm.get_train_summary(),
            self._vm.get_train_result_summary(),
            self._vm.get_valid_result_summary(),
        )
        self._refresh_validation_table(last_epoch, self._vm.get_valid_result_summary())

    def _init_training_chart(self) -> None:
        self._canvas_graph.reset()
        self._canvas_graph_metric.reset()
        self._canvas_graph.setName("Train Loss")
        self._canvas_graph.setBottomName("Epoch")
        self._canvas_graph.setLine("Loss", [], QtGui.QColor(205, 92, 92), 2)

        self._canvas_graph_metric.setName(f"Validation {self._vm.metric_name}")
        self._canvas_graph_metric.setBottomName("Epoch")
        self._canvas_graph_metric.setLine(self._vm.metric_name, [], QtGui.QColor(92, 205, 92), 2)

    def _init_training_history(self) -> None:
        self.table_training_history.clear()
        h_names = [
            "Train\nLoss", f"Valid\n{self._vm.metric_name}",
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
        worker_train_summary: TrainSummary,
        worker_train_result_summary: ResultSummary,
        worker_valid_result_summary: ResultSummary,
    ) -> None:
        tr_loss = []
        va_metric = []
        for epoch in worker_train_summary.tr_by_epoch:
            tr_loss.append(worker_train_summary.tr_by_epoch[epoch]["loss"])
            if epoch in worker_train_summary.va_by_epoch:
                if epoch in worker_valid_result_summary.map:
                    va_metric.append(worker_valid_result_summary.get_metric(epoch, self._vm.metric_name))
                else:
                    va_metric.append(va_metric[-1] if va_metric else 0)
            else:
                va_metric.append(numpy.nan)

        self._canvas_graph.setName("Train Loss")
        self._canvas_graph.setLine("Loss", tr_loss, QtGui.QColor(205, 92, 92), 2)
        self._canvas_graph_metric.setName(f"Validation {self._vm.metric_name}")
        self._canvas_graph_metric.setLine(self._vm.metric_name, va_metric, QtGui.QColor(92, 205, 92), 2)

    def _refresh_training_history_table(
        self,
        worker_train_summary: TrainSummary,
        worker_train_result_summary: ResultSummary,
        worker_valid_result_summary: ResultSummary,
    ) -> None:
        self.table_training_history.clear()
        h_names = [
            "Train\nLoss", f"Valid\n{self._vm.metric_name}",
            "Corner\nError", "Corner\nX", "Corner\nY",
            "Center\nError", "Center\nX", "Center\nY",
            "Longside\nError", "Shortside\nError", "Update",
        ]
        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(len(worker_train_summary.tr_by_epoch))

        header_epoch_map = {}
        for idx, (ep, info) in enumerate(worker_train_summary.tr_by_epoch.items()):
            self.table_training_history.setVerticalHeaderItem(idx, QTableWidgetItem(str(ep)))
            self.table_training_history.setItem(idx, 0, QTableWidgetItem("%.4lf" % info["loss"]))
            header_epoch_map[ep] = idx

        for ep, info in worker_train_summary.va_by_epoch.items():
            idx = header_epoch_map[ep]
            self.table_training_history.setItem(
                idx, 1, QTableWidgetItem("%.4lf" % worker_valid_result_summary.get_metric(ep, self._vm.metric_name))
            )
            mPE = worker_valid_result_summary.get_metric(ep, "mPE")
            for col, key in enumerate(
                ["corner_error", "corner_dx", "corner_dy", "center_error", "center_dx", "center_dy", "longside", "shortside"],
                start=2,
            ):
                self.table_training_history.setItem(idx, col, QTableWidgetItem(f"{mPE[key]:.1f}"))

        for updated_epoch in worker_train_summary.update_model_epoch:
            idx = header_epoch_map[updated_epoch]
            self.table_training_history.setItem(idx, 10, QTableWidgetItem("O"))

        self.table_training_history.resizeColumnsToContents()
        self.table_training_history.resizeRowsToContents()
        self.table_training_history.scrollToBottom()

    def _refresh_validation_table(self, epoch: int, worker_valid_result_summary: ResultSummary) -> None:
        self.table_validation.clear()
        self.table_validation.clearSpans()
        n_class = len(worker_valid_result_summary.class_name)
        if n_class <= 0:
            return

        h_class_names = [worker_valid_result_summary.class_name[i] for i in range(n_class)]
        self.table_validation.setColumnCount(len(h_class_names))
        self.table_validation.setHorizontalHeaderLabels(h_class_names)
        v_class_names = [
            "#", "AP", "Epoch", self._vm.metric_name,
            "Corner Error", "Corner X", "Corner Y",
            "Center Error", "Center X", "Center Y",
            "Width Error", "Height Error",
        ]
        self.table_validation.setRowCount(len(v_class_names))
        self.table_validation.setVerticalHeaderLabels(v_class_names)

        def format_metric(value):
            if value is None:
                return "-"
            try:
                if numpy.isnan(value):
                    return "-"
            except TypeError:
                pass
            return f"{value:.1f}"

        if epoch in worker_valid_result_summary.aps:
            aps = worker_valid_result_summary.aps[epoch]
            if aps is not None:
                for c in range(n_class):
                    item_count = QTableWidgetItem(str(int(aps[c][1])))
                    item_count.setTextAlignment(Qt.AlignCenter)
                    item_count.setBackground(QtGui.QColor(144, 164, 174))
                    self.table_validation.setItem(0, c, item_count)

                    item_ap = QTableWidgetItem("%.2lf" % aps[c][0])
                    item_ap.setTextAlignment(Qt.AlignCenter)
                    item_ap.setBackground(QtGui.QColor(144, 164, 174))
                    self.table_validation.setItem(1, c, item_ap)

        def set_overall_row(row_no: int, text: str) -> None:
            self.table_validation.setSpan(row_no, 0, 1, n_class)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QtGui.QColor(65, 78, 92))
            self.table_validation.setItem(row_no, 0, item)

        set_overall_row(2, str(epoch))

        if epoch in worker_valid_result_summary.map:
            set_overall_row(3, "%.4lf" % worker_valid_result_summary.map[epoch])

        if epoch in worker_valid_result_summary.mpe:
            mpe = worker_valid_result_summary.get_metric(epoch, "mPE")
            class_metrics = worker_valid_result_summary.mpe_by_class.get(epoch, {})
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

    # ------------------------------------------------------------------
    # QWidget lifecycle
    # ------------------------------------------------------------------

    def show(self) -> None:
        super().show()
        self._vm.refresh_resume_state()

        train_summary = self._vm.get_train_summary()
        if not train_summary.tr_by_epoch:
            return
        try:
            last_epoch = list(train_summary.tr_by_epoch.keys())[-1]
            self._refresh_training_chart(
                train_summary,
                self._vm.get_train_result_summary(),
                self._vm.get_valid_result_summary(),
            )
            self._refresh_training_history_table(
                train_summary,
                self._vm.get_train_result_summary(),
                self._vm.get_valid_result_summary(),
            )
            self._refresh_validation_table(last_epoch, self._vm.get_valid_result_summary())
        except Exception:
            pass

    def close(self) -> bool:
        self._vm.close()
        return super().close()
