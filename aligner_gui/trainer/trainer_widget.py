from __future__ import annotations

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from aligner_gui.ui.trainer_widget import Ui_trainer_widget
import logging
import time
from aligner_gui.trainer.thread_train import ThreadTrain
from aligner_gui.trainer.training_timer import TrainingTimer, timestamp2time
import traceback
from aligner_gui.utils import const
from aligner_gui.utils import gui_util
from aligner_gui.widgets.graph_widget import GraphWidget
from aligner_gui.widgets.progress_general_dialog import ProgressGeneralDialog
from aligner_gui.project.project_dataset_service import build_dataset_summary_from_project
from aligner_gui.trainer.gpu_monitor import GpuUsageSnapshot, NvidiaSmiPoller
from aligner_engine.project_settings import ProjectSettings
from aligner_engine.summary import TrainSummary, ResultSummary
from aligner_engine.const import PHASE_TYPE_TRAINING, PHASE_TYPE_VALIDATION
import numpy
from typing import Callable

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class TrainerWidget(QWidget, Ui_trainer_widget):
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

    def __init__(self, main_window, session, tester_reload_callback: Callable[[], None] | None = None):
        super().__init__()
        self.setupUi(self)
        from aligner_gui.main_window import MainWindow
        self._main_window: MainWindow = main_window
        self._worker = session
        self._tester_reload_callback = tester_reload_callback

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

        self.device_panel = QFrame()
        self.device_panel.setMinimumWidth(120)
        self.device_panel.setMaximumWidth(160)
        self.device_panel.setFrameShape(QFrame.StyledPanel)
        self.device_panel.setFrameShadow(QFrame.Raised)
        self.device_panel_layout = QVBoxLayout(self.device_panel)
        self.device_panel_layout.setContentsMargins(8, 8, 8, 8)
        self.device_panel_layout.setSpacing(6)
        self.device_panel_layout.setAlignment(Qt.AlignTop)

        self.label_device_title = QLabel("GPU Mem")
        self.label_device_title.setAlignment(Qt.AlignCenter)
        self.device_panel_layout.addWidget(self.label_device_title)

        self.progress_device_usage = QProgressBar()
        self.progress_device_usage.setOrientation(Qt.Vertical)
        self.progress_device_usage.setFixedSize(20, 118)
        self.progress_device_usage.setRange(0, 100)
        self.progress_device_usage.setTextVisible(False)
        self.progress_device_usage.setInvertedAppearance(False)
        self.progress_device_usage.setStyleSheet(self.DEVICE_BAR_STYLE)
        self.progress_device_usage.hide()
        self.device_panel_layout.addWidget(self.progress_device_usage, alignment=Qt.AlignHCenter)

        self.label_device_percent = QLabel("")
        self.label_device_percent.setAlignment(Qt.AlignCenter)
        self.label_device_percent.setStyleSheet("font-weight: 600; color: rgb(220, 226, 232);")
        self.device_panel_layout.addWidget(self.label_device_percent)

        self.label_runtime_info = QLabel("VRAM --/--")
        self.label_runtime_info.setAlignment(Qt.AlignCenter)
        self.label_runtime_info.setStyleSheet("color: rgb(170, 180, 190);")
        self.label_runtime_info.setWordWrap(False)
        self.device_panel_layout.addWidget(self.label_runtime_info)
        self.device_panel_layout.addStretch(1)

        self.layout_visualization.addWidget(self.device_panel)
        self._init_model_profile_ui()

        # bind handler
        self.btn_train.clicked.connect(self._clicked_btn_train)
        self.check_hflip.clicked.connect(self._clicked_check_hflip)
        self.check_vflip.clicked.connect(self._clicked_check_vflip)
        self.check_no_rotation.clicked.connect(self._clicked_check_no_rotation)
        self.check_include_empty.clicked.connect(self._clicked_check_include_empty)
        self.spin_resize.valueChanged.connect(self._changed_spin_resize)
        self.spin_batch_size.valueChanged.connect(self._changed_spin_batch_size)
        self.spin_max_epochs.valueChanged.connect(self._changed_max_epochs)
        self.table_training_history.clicked.connect(self._clicked_table_training_history)

        # Ui init
        settings = self._get_settings()
        self.check_hflip.setChecked(settings.aug_flip_horizontal_use)
        self.check_vflip.setChecked(settings.aug_flip_vertical_use)
        self.check_no_rotation.setChecked(settings.no_rotation)
        self.check_include_empty.setChecked(settings.include_empty)
        self.spin_resize.setValue(settings.resize)
        self.spin_batch_size.setValue(settings.batch_size)
        self.spin_max_epochs.setValue(settings.max_epochs)

        # thread setting
        self._th_train = ThreadTrain(self._worker)
        self._th_train.qt_signal_stop_training.connect(self._stop_training)
        self._th_train.qt_signal_update_epoch.connect(self._update_epoch_th_train)
        self._th_train.qt_signal_update_iter.connect(self._update_iter_th_train)
        self._th_train.qt_signal_status.connect(self._update_training_status)

        # for training
        self._busy_indicator = QtGui.QMovie("aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif")
        self.lbl_train_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_train_indicator.hide()

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
        self._refresh_resume_ui()

    def _init_model_profile_ui(self):
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

        settings = self._get_settings()
        current_profile = settings.model_profile
        for profile in self._worker.get_model_profiles():
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
                settings.model_profile = selected_profile_id
                self._save_settings(settings)
        self.combo_model_profile.setEnabled(False)
        self.combo_model_profile.setToolTip("Model profile switching is temporarily disabled until the corresponding pretrained weights are bundled.")
        self.lbl_model_profile.hide()
        self.combo_model_profile.hide()

    def _get_settings(self) -> ProjectSettings:
        return self._worker.get_project_settings()

    def _save_settings(self, settings: ProjectSettings):
        self._worker.set_project_settings(settings)

    def _clicked_check_hflip(self):
        settings = self._get_settings()
        settings.aug_flip_horizontal_use = self.check_hflip.isChecked()
        self._save_settings(settings)

    def _clicked_check_vflip(self):
        settings = self._get_settings()
        settings.aug_flip_vertical_use = self.check_vflip.isChecked()
        self._save_settings(settings)

    def _clicked_check_no_rotation(self):
        settings = self._get_settings()
        settings.no_rotation = self.check_no_rotation.isChecked()
        self._save_settings(settings)

    def _clicked_check_include_empty(self):
        settings = self._get_settings()
        settings.include_empty = self.check_include_empty.isChecked()
        self._save_settings(settings)

    def _changed_spin_resize(self):
        settings = self._get_settings()
        settings.resize = self.spin_resize.value()
        self._save_settings(settings)

    def _changed_max_epochs(self):
        settings = self._get_settings()
        settings.max_epochs = self.spin_max_epochs.value()
        self._save_settings(settings)

    def _changed_spin_batch_size(self):
        settings = self._get_settings()
        settings.batch_size = self.spin_batch_size.value()
        self._save_settings(settings)

    def _changed_model_profile(self):
        if not self.combo_model_profile.isEnabled():
            return
        profile_id = self.combo_model_profile.currentData()
        if not profile_id:
            return
        settings = self._get_settings()
        if settings.model_profile == profile_id:
            return
        settings.model_profile = profile_id
        self._save_settings(settings)

    def _get_torch(self):
        import torch

        return torch

    def _prepare_training_assets(self) -> bool:
        settings = self._get_settings()

        def work(processing_signal):
            processing_signal.emit(1, "Building dataset summary...")
            build_dataset_summary_from_project(
                self._worker.project_path,
                self._worker.get_dataset_summary_path(),
                settings.include_empty,
            )
            processing_signal.emit(2, "Preparing training workspace...")
            return True

        dlg = ProgressGeneralDialog("Preparing training...", work, 2)
        dlg.exec_()
        if not dlg.is_success:
            gui_util.get_message_box(
                self,
                "Invalid Dataset",
                "Failed to prepare the dataset.\nChoose images in the labeler first and make sure enough labeled data exists.",
            )
            return False

        if self._tester_reload_callback is not None:
            self._tester_reload_callback()
        return True

    def _refresh_resume_ui(self):
        can_resume = self._worker.can_resume_training()
        last_epoch = self._worker.get_last_completed_epoch()
        self.check_resume.setEnabled(can_resume)
        if not can_resume:
            self.check_resume.setChecked(False)
            self.check_resume.setToolTip("Enable after at least one checkpoint has been saved.")
            return
        resume_epoch = last_epoch + 1 if last_epoch > 0 else 1
        self.check_resume.setToolTip(
            f"Continue training from auto_saved\\last.pth.\nNext epoch: {resume_epoch}"
        )

    def _refresh_existing_training_monitor(self, last_epoch: int):
        self._refresh_training_chart(
            self._worker.get_train_summary(),
            self._worker.get_train_result_summary(),
            self._worker.get_valid_result_summary(),
        )
        self._refresh_training_history_table(
            self._worker.get_train_summary(),
            self._worker.get_train_result_summary(),
            self._worker.get_valid_result_summary(),
        )
        self._refresh_validation_table(last_epoch, self._worker.get_valid_result_summary())

    def _start_training(self):
        try:
            TRAIN_LOGGER.info('start training')
            resume_training = self.check_resume.isChecked()
            if resume_training and not self._worker.can_resume_training():
                self._refresh_resume_ui()
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "No last checkpoint was found.\nRun at least one epoch first or start a fresh training.",
                )
                return
            if resume_training and self._worker.get_last_completed_epoch() >= self._get_settings().max_epochs:
                gui_util.get_message_box(
                    self,
                    "Resume Unavailable",
                    "The latest checkpoint already reached the configured max epochs.\nIncrease Epochs to continue training.",
                )
                return
            self._on_start_training(resume_training)
            if not self._prepare_training_assets():
                self._stop_training(const.ERROR)
                return
            TRAIN_LOGGER.info("Dataset is built successfully")
            torch = self._get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._th_train.set_resume_training(resume_training)
            self._th_train.start()

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            TRAIN_LOGGER.error(error_msg)
            self._stop_training(const.ERROR)

    def _stop_training(self, reason: str):
        if self._th_train.isRunning():
            self._th_train.terminate()

        torch = self._get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reason == const.SUCCESS:
            TRAIN_LOGGER.info("training finished successfully")
        elif reason == const.ERROR:
            TRAIN_LOGGER.error("training failed")
        else:
            TRAIN_LOGGER.info(reason)
        self._on_stop_training()

    def _on_start_training(self, resume_training: bool):
        settings = self._get_settings()
        last_epoch = self._worker.get_last_completed_epoch() if resume_training else 0

        self.btn_train.setChecked(True)
        self.btn_train.setText("Stop")
        self.btn_train.setEnabled(True)

        if resume_training and last_epoch > 0:
            self.label_time.setText(f"Resuming from epoch {last_epoch + 1}...")
        else:
            self.label_time.setText("Preparing training...")
        self.label_device_title.setText("GPU Mem")
        self.label_device_percent.setText("")
        self.label_runtime_info.setText("VRAM --/--")
        if resume_training and last_epoch > 0:
            self._refresh_existing_training_monitor(last_epoch)
        else:
            self._init_training_chart()
            self._init_training_history()
            self._init_validation_table()


        self.progress_epoch.setMaximum(settings.max_epochs)
        self.progress_epoch.setValue(last_epoch)
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

        self._training_timer.train_start(last_epoch, settings.max_epochs)
        self._current_epoch_for_eta = max(last_epoch + 1, 1)
        self._last_iter_ui_update_ts = 0.0
        self._gpu_monitor_available = None
        self.progress_device_usage.setValue(0)
        self.progress_device_usage.show()
        torch = self._get_torch()
        if torch.cuda.is_available():
            self._gpu_monitor.set_preferred_device_index(torch.cuda.current_device())
        self._device_poll_timer.start()
        self._refresh_device_usage()
        self._main_window.set_app_status_training()

    def _on_stop_training(self):
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
        self._refresh_resume_ui()
        self._device_poll_timer.stop()
        self._gpu_monitor.stop()
        self._gpu_monitor_available = None
        self.progress_device_usage.hide()
        self.progress_device_usage.setValue(0)
        self.label_device_percent.setText("")
        self.label_device_title.setText("GPU Mem")
        self.label_runtime_info.setText("VRAM --/--")
        self._main_window.set_app_status_idle()

    def _update_training_status(self, message: str):
        if self.progress_epoch.value() == 0 and self.progress_iter.value() == 0:
            self.label_time.setText(message)

    def _clicked_btn_train(self):
        if self.btn_train.isChecked():
            self._start_training()
        else:
            self._stop_training("The process has been terminated at the user's request.")

    def close(self):
        self._stop_training("window close")
        return super().close()

    def _refresh_training_time(self, cur_epoch):
        one_epoch, avg_one_epoch, processed, remaining = self._training_timer.one_epoch_done(cur_epoch)
        estTimeMsg = 'Avg epoch %.1fs   Elapsed %s   Remaining %s' \
                     % (avg_one_epoch, timestamp2time(processed), timestamp2time(remaining))
        self.label_time.setText(estTimeMsg)

    def _refresh_training_time_live(self, phase_type: str, iter_idx: int, iter_len: int):
        current_epoch = min(max(self.progress_epoch.value() + 1, 1), max(self.progress_epoch.maximum(), 1))
        self._current_epoch_for_eta = current_epoch
        avg_step, processed, remaining = self._training_timer.one_iter_progress(
            phase_type,
            iter_idx,
            iter_len,
            current_epoch,
        )
        phase_name = "Train" if phase_type == PHASE_TYPE_TRAINING else "Valid"
        self.label_time.setText(
            f'{phase_name} {iter_idx + 1}/{iter_len}   Avg step {avg_step:.2f}s   Elapsed {timestamp2time(processed)}   Remaining {timestamp2time(remaining)}'
        )

    def _sample_gpu_memory_stats(self):
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
            "text": f'{used_memory / (1024 ** 3):.1f}/{total_memory / (1024 ** 3):.1f} GB ({usage:.0f}%)',
        }

    def _refresh_device_usage(self):
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

    def _apply_gpu_snapshot(self, snapshot: GpuUsageSnapshot):
        self._gpu_monitor_available = True
        memory_percent = max(0.0, min(snapshot.memory_percent, 100.0))
        used_gb = snapshot.memory_used_mb / 1024.0
        total_gb = snapshot.memory_total_mb / 1024.0
        self.progress_device_usage.show()
        self.progress_device_usage.setValue(int(round(memory_percent)))
        self.label_device_title.setText("GPU Mem")
        self.label_device_percent.setText(f'{memory_percent:.0f}%')
        self.label_runtime_info.setText(f'{used_gb:.1f}/{total_gb:.1f} GB ({memory_percent:.0f}%)')
        self.label_runtime_info.setToolTip(snapshot.name)

    def _handle_gpu_monitor_unavailable(self, _: str):
        self._gpu_monitor_available = False
        self._apply_gpu_memory_fallback()

    def _apply_gpu_memory_fallback(self):
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

    def _update_epoch_th_train(self, current_epoch: int, current_ckpt_path: str):
        self.progress_epoch.setValue(current_epoch)
        self._current_epoch_for_eta = current_epoch + 1
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self._worker.save_records_after_epoch(current_epoch, current_ckpt_path)

        self._refresh_training_time(current_epoch)
        self._refresh_training_chart(self._worker.get_train_summary(), self._worker.get_train_result_summary(),
                                     self._worker.get_valid_result_summary())
        self._refresh_training_history_table(self._worker.get_train_summary(), self._worker.get_train_result_summary(),
                                             self._worker.get_valid_result_summary())
        self._refresh_validation_table(current_epoch, self._worker.get_valid_result_summary())

    def _update_iter_th_train(self, phase_type: str, iter_idx: int, iter_len: int):
        if phase_type == PHASE_TYPE_TRAINING:
            self.progress_iter.setStyleSheet(TrainerWidget.COLOR_TRAINING)
        elif phase_type == PHASE_TYPE_VALIDATION:
            self.progress_iter.setStyleSheet(TrainerWidget.COLOR_VALIDATION)
        else:
            raise NotImplementedError

        self.progress_iter.setMaximum(iter_len)
        self.progress_iter.setValue(iter_idx + 1)
        now = time.monotonic()
        is_last_iter = iter_len > 0 and (iter_idx + 1) >= iter_len
        if is_last_iter or (now - self._last_iter_ui_update_ts) >= 0.15:
            self._last_iter_ui_update_ts = now
            self._refresh_training_time_live(phase_type, iter_idx, iter_len)

    def show(self):
        super().show()
        self._refresh_resume_ui()

        if len(self._worker.get_train_summary().tr_by_epoch.keys()) == 0:
            return

        try:
            last_epoch = list(self._worker.get_train_summary().tr_by_epoch.keys())[-1]
            self._refresh_training_chart(self._worker.get_train_summary(), self._worker.get_train_result_summary(),
                                         self._worker.get_valid_result_summary())
            self._refresh_training_history_table(self._worker.get_train_summary(),
                                                 self._worker.get_train_result_summary(),
                                                 self._worker.get_valid_result_summary())
            self._refresh_validation_table(last_epoch, self._worker.get_valid_result_summary())
        except Exception as e:
            pass


    def _init_training_chart(self):
        self._canvas_graph.reset()
        self._canvas_graph_metric.reset()
        self._canvas_graph.setName('Train Loss')
        self._canvas_graph.setBottomName('Epoch')
        self._canvas_graph.setLine('Loss', [], QtGui.QColor(205, 92, 92), 2)

        self._canvas_graph_metric.setName(f'Validation {self._worker.metric_name}')
        self._canvas_graph_metric.setBottomName('Epoch')
        self._canvas_graph_metric.setLine(self._worker.metric_name, [], QtGui.QColor(92, 205, 92), 2)

    def _init_training_history(self):
        self.table_training_history.clear()
        h_names = ['Train\nLoss', 'Valid\n%s' % self._worker.metric_name,
                   'Corner\nError',
                   'Corner\nX',
                   'Corner\nY',
                   'Center\nError',
                   'Center\nX',
                   'Center\nY',
                   'Longside\nError',
                   'Shortside\nError', 'Update']

        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(1)

    def _init_validation_table(self):
        self.table_validation.clear()

    def _refresh_training_chart(self, worker_train_summary: TrainSummary,
                                worker_train_result_summary: ResultSummary,
                                worker_valid_result_summary: ResultSummary):
        tr_loss = []
        va_metric = []
        for epoch in worker_train_summary.tr_by_epoch:
            tr_loss.append(worker_train_summary.tr_by_epoch[epoch]['loss'])

            if epoch in worker_train_summary.va_by_epoch:
                if epoch in worker_valid_result_summary.map:
                    va_metric.append(worker_valid_result_summary.get_metric(epoch, self._worker.metric_name))
                else:
                    if len(va_metric) > 0:
                        va_metric.append(va_metric[-1])
                    else:
                        va_metric.append(0)
            else:
                va_metric.append(numpy.nan)

        self._canvas_graph.setName('Train Loss')
        self._canvas_graph.setLine('Loss', tr_loss, QtGui.QColor(205, 92, 92), 2)

        self._canvas_graph_metric.setName(f'Validation {self._worker.metric_name}')
        self._canvas_graph_metric.setLine(self._worker.metric_name, va_metric, QtGui.QColor(92, 205, 92), 2)

    def _refresh_training_history_table(self, worker_train_summary: TrainSummary,
                                        worker_train_result_summary: ResultSummary,
                                        worker_valid_result_summary: ResultSummary):
        self.table_training_history.clear()

        h_names = ['Train\nLoss', 'Valid\n%s' % self._worker.metric_name,
                   'Corner\nError',
                   'Corner\nX',
                   'Corner\nY',
                   'Center\nError',
                   'Center\nX',
                   'Center\nY',
                   'Longside\nError',
                   'Shortside\nError', 'Update']
        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(len(worker_train_summary.tr_by_epoch))

        header_epoch_map = {}

        for idx, (ep, info) in enumerate(worker_train_summary.tr_by_epoch.items()):
            self.table_training_history.setVerticalHeaderItem(idx, QTableWidgetItem(str(ep)))
            self.table_training_history.setItem(idx, 0, QTableWidgetItem('%.4lf' % info['loss']))
            header_epoch_map[ep] = idx

        for ep, info in worker_train_summary.va_by_epoch.items():
            idx = header_epoch_map[ep]
            self.table_training_history.setItem(idx, 1, QTableWidgetItem(
                '%.4lf' % worker_valid_result_summary.get_metric(ep, self._worker.metric_name)))

            mPE = worker_valid_result_summary.get_metric(ep, 'mPE')
            corner_error = f'{mPE["corner_error"]:.1f}'
            self.table_training_history.setItem(idx, 2, QTableWidgetItem(corner_error))

            corner_dx = f'{mPE["corner_dx"]:.1f}'
            self.table_training_history.setItem(idx, 3, QTableWidgetItem(corner_dx))

            corner_dy = f'{mPE["corner_dy"]:.1f}'
            self.table_training_history.setItem(idx, 4, QTableWidgetItem(corner_dy))

            center_error = f'{mPE["center_error"]:.1f}'
            self.table_training_history.setItem(idx, 5, QTableWidgetItem(center_error))

            center_dx = f'{mPE["center_dx"]:.1f}'
            self.table_training_history.setItem(idx, 6, QTableWidgetItem(center_dx))

            center_dy = f'{mPE["center_dy"]:.1f}'
            self.table_training_history.setItem(idx, 7, QTableWidgetItem(center_dy))

            longside = f'{mPE["longside"]:.1f}'
            self.table_training_history.setItem(idx, 8, QTableWidgetItem(longside))

            shortside = f'{mPE["shortside"]:.1f}'
            self.table_training_history.setItem(idx, 9, QTableWidgetItem(shortside))

        for updated_epoch in worker_train_summary.update_model_epoch:
            idx = header_epoch_map[updated_epoch]
            self.table_training_history.setItem(idx, 10, QTableWidgetItem('O'))

        self.table_training_history.resizeColumnsToContents()
        self.table_training_history.resizeRowsToContents()
        self.table_training_history.scrollToBottom()

    def _refresh_validation_table(self, epoch: int, worker_valid_result_summary: ResultSummary):
        self.table_validation.clear()
        self.table_validation.clearSpans()
        n_class = len(worker_valid_result_summary.class_name)
        if n_class <= 0:
            return

        h_class_names = [worker_valid_result_summary.class_name[i] for i in range(n_class)]
        self.table_validation.setColumnCount(len(h_class_names))
        self.table_validation.setHorizontalHeaderLabels(h_class_names)
        v_class_names = ['#', 'AP', 'Epoch', self._worker.metric_name, 'Corner Error', 'Corner X', 'Corner Y',
                         'Center Error', 'Center X', 'Center Y', 'Width Error', 'Height Error']
        self.table_validation.setRowCount(len(v_class_names))
        self.table_validation.setVerticalHeaderLabels(v_class_names)

        def format_metric(value):
            if value is None:
                return '-'
            try:
                if numpy.isnan(value):
                    return '-'
            except TypeError:
                pass
            return f'{value:.1f}'

        if epoch in worker_valid_result_summary.aps:
            aps = worker_valid_result_summary.aps[epoch]
            if aps is not None:
                for c in range(n_class):
                    self.table_validation.setItem(0, c, QTableWidgetItem(str(int(aps[c][1]))))  # num gts
                    self.table_validation.item(0, c).setTextAlignment(Qt.AlignCenter)
                    self.table_validation.item(0, c).setBackground(QtGui.QColor(144, 164, 174))

                    self.table_validation.setItem(1, c, QTableWidgetItem('%.2lf' % aps[c][0]))  # ap
                    self.table_validation.item(1, c).setTextAlignment(Qt.AlignCenter)
                    self.table_validation.item(1, c).setBackground(QtGui.QColor(144, 164, 174))

        def set_overall_row(row_no: int, text: str):
            self.table_validation.setSpan(row_no, 0, 1, n_class)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QtGui.QColor(65, 78, 92))
            self.table_validation.setItem(row_no, 0, item)

        set_overall_row(2, str(epoch))

        if epoch in worker_valid_result_summary.map:
            map = worker_valid_result_summary.map[epoch]
            set_overall_row(3, '%.4lf' % map)

        if epoch in worker_valid_result_summary.mpe:
            mpe = worker_valid_result_summary.get_metric(epoch, 'mPE')
            class_metrics = worker_valid_result_summary.mpe_by_class.get(epoch, {})
            if class_metrics:
                row_key_map = {
                    4: 'corner_error',
                    5: 'corner_dx',
                    6: 'corner_dy',
                    7: 'center_error',
                    8: 'center_dx',
                    9: 'center_dy',
                    10: 'longside',
                    11: 'shortside',
                }
                for row_no, metric_key in row_key_map.items():
                    for c in range(n_class):
                        value = class_metrics.get(c, {}).get(metric_key)
                        item = QTableWidgetItem(format_metric(value))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table_validation.setItem(row_no, c, item)
            else:
                set_overall_row(4, f'{mpe["corner_error"]:.1f}')
                set_overall_row(5, f'{mpe["corner_dx"]:.1f}')
                set_overall_row(6, f'{mpe["corner_dy"]:.1f}')
                set_overall_row(7, f'{mpe["center_error"]:.1f}')
                set_overall_row(8, f'{mpe["center_dx"]:.1f}')
                set_overall_row(9, f'{mpe["center_dy"]:.1f}')
                set_overall_row(10, f'{mpe["longside"]:.1f}')
                set_overall_row(11, f'{mpe["shortside"]:.1f}')

        self.table_validation.resizeColumnsToContents()
        self.table_validation.resizeRowsToContents()

    def _clicked_table_training_history(self):
        epoch = self.table_training_history.currentRow() + 1
        self._refresh_validation_table(epoch, self._worker.get_valid_result_summary())
