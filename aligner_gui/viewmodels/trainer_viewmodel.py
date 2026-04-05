from __future__ import annotations

import logging
import time
import traceback
from typing import Callable, TYPE_CHECKING

from PyQt5.QtCore import QTimer

from aligner_gui.project.project_dataset_service import build_dataset_summary_from_project
from aligner_gui.trainer.gpu_monitor import GpuUsageSnapshot, NvidiaSmiPoller
from aligner_gui.trainer.thread_train import ThreadTrain
from aligner_gui.trainer.training_timer import TrainingTimer, timestamp2time
from aligner_gui.shared import const, gui_util
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase
from aligner_gui.shared.progress_general_dialog import ProgressGeneralDialog
from aligner_engine.const import PHASE_TYPE_TRAINING, PHASE_TYPE_VALIDATION
from aligner_engine.project_settings import ProjectSettings

if TYPE_CHECKING:
    from aligner_gui.trainer.trainer_view import TrainerView

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class TrainerViewModel(ViewModelBase):
    def __init__(self, view: 'TrainerView', session, tester_reload_callback: Callable[[], None] | None = None):
        super().__init__(view)
        self.view = view
        self._worker = session
        self._tester_reload_callback = tester_reload_callback
        self._th_train = ThreadTrain(self._worker)
        self._training_timer = TrainingTimer()
        self._device_poll_timer = QTimer(view)
        self._device_poll_timer.setInterval(1500)
        self._device_poll_timer.timeout.connect(self.refresh_device_usage)
        self._gpu_monitor = NvidiaSmiPoller(view)
        self._gpu_monitor.stats_updated.connect(self.apply_gpu_snapshot)
        self._gpu_monitor.unavailable.connect(self.handle_gpu_monitor_unavailable)
        self._gpu_monitor_available: bool | None = None
        self._current_epoch_for_eta = 1
        self._last_iter_ui_update_ts = 0.0

    def initialize(self):
        self.view._init_model_profile_ui()
        self.view.btn_train.clicked.connect(self.handle_train_button_clicked)
        self.view.check_hflip.clicked.connect(self.clicked_check_hflip)
        self.view.check_vflip.clicked.connect(self.clicked_check_vflip)
        self.view.check_no_rotation.clicked.connect(self.clicked_check_no_rotation)
        self.view.check_include_empty.clicked.connect(self.clicked_check_include_empty)
        self.view.spin_resize.valueChanged.connect(self.changed_spin_resize)
        self.view.spin_batch_size.valueChanged.connect(self.changed_spin_batch_size)
        self.view.spin_max_epochs.valueChanged.connect(self.changed_max_epochs)
        self.view.table_training_history.clicked.connect(self.view._clicked_table_training_history)

        settings = self.get_settings()
        self.view.check_hflip.setChecked(settings.aug_flip_horizontal_use)
        self.view.check_vflip.setChecked(settings.aug_flip_vertical_use)
        self.view.check_no_rotation.setChecked(settings.no_rotation)
        self.view.check_include_empty.setChecked(settings.include_empty)
        self.view.spin_resize.setValue(settings.resize)
        self.view.spin_batch_size.setValue(settings.batch_size)
        self.view.spin_max_epochs.setValue(settings.max_epochs)

        self._th_train.qt_signal_stop_training.connect(self.stop_training)
        self._th_train.qt_signal_update_epoch.connect(self.update_epoch)
        self._th_train.qt_signal_update_iter.connect(self.update_iter)
        self._th_train.qt_signal_status.connect(self.update_training_status)
        self.refresh_resume_ui()

    def get_settings(self) -> ProjectSettings:
        return self._worker.get_project_settings()

    def save_settings(self, settings: ProjectSettings):
        self._worker.set_project_settings(settings)

    def clicked_check_hflip(self):
        settings = self.get_settings()
        settings.aug_flip_horizontal_use = self.view.check_hflip.isChecked()
        self.save_settings(settings)

    def clicked_check_vflip(self):
        settings = self.get_settings()
        settings.aug_flip_vertical_use = self.view.check_vflip.isChecked()
        self.save_settings(settings)

    def clicked_check_no_rotation(self):
        settings = self.get_settings()
        settings.no_rotation = self.view.check_no_rotation.isChecked()
        self.save_settings(settings)

    def clicked_check_include_empty(self):
        settings = self.get_settings()
        settings.include_empty = self.view.check_include_empty.isChecked()
        self.save_settings(settings)

    def changed_spin_resize(self):
        settings = self.get_settings()
        settings.resize = self.view.spin_resize.value()
        self.save_settings(settings)

    def changed_max_epochs(self):
        settings = self.get_settings()
        settings.max_epochs = self.view.spin_max_epochs.value()
        self.save_settings(settings)

    def changed_spin_batch_size(self):
        settings = self.get_settings()
        settings.batch_size = self.view.spin_batch_size.value()
        self.save_settings(settings)

    def changed_model_profile(self):
        if not self.view.combo_model_profile.isEnabled():
            return
        profile_id = self.view.combo_model_profile.currentData()
        if not profile_id:
            return
        settings = self.get_settings()
        if settings.model_profile == profile_id:
            return
        settings.model_profile = profile_id
        self.save_settings(settings)

    def get_torch(self):
        import torch
        return torch

    def prepare_training_assets(self) -> bool:
        settings = self.get_settings()

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
                self.view,
                "Invalid Dataset",
                "Failed to prepare the dataset.\nChoose images in the labeler first and make sure enough labeled data exists.",
            )
            return False

        if self._tester_reload_callback is not None:
            self._tester_reload_callback()
        return True

    def refresh_resume_ui(self):
        can_resume = self._worker.can_resume_training()
        last_epoch = self._worker.get_last_completed_epoch()
        self.view.check_resume.setEnabled(can_resume)
        if not can_resume:
            self.view.check_resume.setChecked(False)
            self.view.check_resume.setToolTip("Enable after at least one checkpoint has been saved.")
            return
        resume_epoch = last_epoch + 1 if last_epoch > 0 else 1
        self.view.check_resume.setToolTip(
            f"Continue training from auto_saved\\last.pth.\nNext epoch: {resume_epoch}"
        )

    def refresh_existing_training_monitor(self, last_epoch: int):
        self.view._refresh_training_chart(
            self._worker.get_train_summary(),
            self._worker.get_train_result_summary(),
            self._worker.get_valid_result_summary(),
        )
        self.view._refresh_training_history_table(
            self._worker.get_train_summary(),
            self._worker.get_train_result_summary(),
            self._worker.get_valid_result_summary(),
        )
        self.view._refresh_validation_table(last_epoch, self._worker.get_valid_result_summary())

    def start_training(self):
        try:
            self.emit_status("Starting training...")
            TRAIN_LOGGER.info('start training')
            resume_training = self.view.check_resume.isChecked()
            if resume_training and not self._worker.can_resume_training():
                self.refresh_resume_ui()
                gui_util.get_message_box(
                    self.view,
                    "Resume Unavailable",
                    "No last checkpoint was found.\nRun at least one epoch first or start a fresh training.",
                )
                self.view.btn_train.setChecked(False)
                return
            if resume_training and self._worker.get_last_completed_epoch() >= self.get_settings().max_epochs:
                gui_util.get_message_box(
                    self.view,
                    "Resume Unavailable",
                    "The latest checkpoint already reached the configured max epochs.\nIncrease Epochs to continue training.",
                )
                self.view.btn_train.setChecked(False)
                return
            self.on_start_training(resume_training)
            if not self.prepare_training_assets():
                self.stop_training(const.ERROR)
                return
            TRAIN_LOGGER.info("Dataset is built successfully")
            torch = self.get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._th_train.set_resume_training(resume_training)
            self._th_train.start()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            TRAIN_LOGGER.error("ERROR - %s", e)
            self.stop_training(const.ERROR)

    def stop_training(self, reason: str):
        if self._th_train.isRunning():
            self._th_train.terminate()

        torch = self.get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reason == const.SUCCESS:
            TRAIN_LOGGER.info("training finished successfully")
            self.emit_status("Training finished successfully.", 3000)
        elif reason == const.ERROR:
            TRAIN_LOGGER.error("training failed")
            self.emit_status("Training failed.", 3000)
        else:
            TRAIN_LOGGER.info(reason)
            self.emit_status(reason, 3000)
        self.on_stop_training()

    def on_start_training(self, resume_training: bool):
        settings = self.get_settings()
        last_epoch = self._worker.get_last_completed_epoch() if resume_training else 0

        self.view.btn_train.setChecked(True)
        self.view.btn_train.setText("Stop")
        self.view.btn_train.setEnabled(True)

        if resume_training and last_epoch > 0:
            self.view.label_time.setText(f"Resuming from epoch {last_epoch + 1}...")
        else:
            self.view.label_time.setText("Preparing training...")
        self.view.label_device_title.setText("GPU Mem")
        self.view.label_device_percent.setText("")
        self.view.label_runtime_info.setText("VRAM --/--")
        if resume_training and last_epoch > 0:
            self.refresh_existing_training_monitor(last_epoch)
        else:
            self.view._init_training_chart()
            self.view._init_training_history()
            self.view._init_validation_table()

        self.view.progress_epoch.setMaximum(settings.max_epochs)
        self.view.progress_epoch.setValue(last_epoch)
        self.view.progress_epoch.setStyleSheet(gui_util.get_dark_style())

        self.view.progress_iter.setMaximum(1)
        self.view.progress_iter.setValue(0)
        self.view.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.view.lbl_train_indicator.show()

        self.view.check_hflip.setEnabled(False)
        self.view.check_vflip.setEnabled(False)
        self.view.check_no_rotation.setEnabled(False)
        self.view.check_include_empty.setEnabled(False)
        self.view.check_resume.setEnabled(False)
        self.view.combo_model_profile.setEnabled(False)
        self.view.spin_resize.setEnabled(False)
        self.view.spin_batch_size.setEnabled(False)
        self.view.spin_max_epochs.setEnabled(False)

        self._training_timer.train_start(last_epoch, settings.max_epochs)
        self._current_epoch_for_eta = max(last_epoch + 1, 1)
        self._last_iter_ui_update_ts = 0.0
        self._gpu_monitor_available = None
        self.view.progress_device_usage.setValue(0)
        self.view.progress_device_usage.show()
        torch = self.get_torch()
        if torch.cuda.is_available():
            self._gpu_monitor.set_preferred_device_index(torch.cuda.current_device())
        self._device_poll_timer.start()
        self.refresh_device_usage()
        self.view._main_window.set_app_status_training()

    def on_stop_training(self):
        self.view.btn_train.setChecked(False)
        self.view.btn_train.setText("Train")
        self.view.btn_train.setEnabled(True)
        self.view.progress_epoch.setMaximum(1)
        self.view.progress_epoch.setValue(0)
        self.view.progress_epoch.setStyleSheet(gui_util.get_dark_style())
        self.view.progress_iter.setMaximum(1)
        self.view.progress_iter.setValue(0)
        self.view.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.view.lbl_train_indicator.hide()
        self.view.check_hflip.setEnabled(True)
        self.view.check_vflip.setEnabled(True)
        self.view.check_no_rotation.setEnabled(True)
        self.view.check_include_empty.setEnabled(True)
        self.view.combo_model_profile.setEnabled(False)
        self.view.spin_resize.setEnabled(True)
        self.view.spin_batch_size.setEnabled(True)
        self.view.spin_max_epochs.setEnabled(True)
        self.refresh_resume_ui()
        self._device_poll_timer.stop()
        self._gpu_monitor.stop()
        self._gpu_monitor_available = None
        self.view.progress_device_usage.hide()
        self.view.progress_device_usage.setValue(0)
        self.view.label_device_percent.setText("")
        self.view.label_device_title.setText("GPU Mem")
        self.view.label_runtime_info.setText("VRAM --/--")
        self.view._main_window.set_app_status_idle()

    def update_training_status(self, message: str):
        if self.view.progress_epoch.value() == 0 and self.view.progress_iter.value() == 0:
            self.view.label_time.setText(message)

    def handle_train_button_clicked(self):
        if self.view.btn_train.isChecked():
            self.start_training()
        else:
            self.stop_training("The process has been terminated at the user's request.")

    def refresh_training_time(self, cur_epoch):
        _one_epoch, avg_one_epoch, processed, remaining = self._training_timer.one_epoch_done(cur_epoch)
        self.view.label_time.setText(
            'Avg epoch %.1fs   Elapsed %s   Remaining %s' % (avg_one_epoch, timestamp2time(processed), timestamp2time(remaining))
        )

    def refresh_training_time_live(self, phase_type: str, iter_idx: int, iter_len: int):
        current_epoch = min(max(self.view.progress_epoch.value() + 1, 1), max(self.view.progress_epoch.maximum(), 1))
        self._current_epoch_for_eta = current_epoch
        avg_step, processed, remaining = self._training_timer.one_iter_progress(
            phase_type, iter_idx, iter_len, current_epoch,
        )
        phase_name = "Train" if phase_type == PHASE_TYPE_TRAINING else "Valid"
        self.view.label_time.setText(
            f'{phase_name} {iter_idx + 1}/{iter_len}   Avg step {avg_step:.2f}s   Elapsed {timestamp2time(processed)}   Remaining {timestamp2time(remaining)}'
        )

    def sample_gpu_memory_stats(self):
        torch = self.get_torch()
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

    def refresh_device_usage(self):
        torch = self.get_torch()
        if not torch.cuda.is_available():
            self.view.progress_device_usage.hide()
            self.view.label_device_title.setText("CPU")
            self.view.label_device_percent.setText("")
            self.view.label_runtime_info.setText("Device: CPU training")
            return
        if self._gpu_monitor_available is not True:
            self.apply_gpu_memory_fallback()
        self._gpu_monitor.poll()

    def apply_gpu_snapshot(self, snapshot: GpuUsageSnapshot):
        self._gpu_monitor_available = True
        memory_percent = max(0.0, min(snapshot.memory_percent, 100.0))
        used_gb = snapshot.memory_used_mb / 1024.0
        total_gb = snapshot.memory_total_mb / 1024.0
        self.view.progress_device_usage.show()
        self.view.progress_device_usage.setValue(int(round(memory_percent)))
        self.view.label_device_title.setText("GPU Mem")
        self.view.label_device_percent.setText(f'{memory_percent:.0f}%')
        self.view.label_runtime_info.setText(f'{used_gb:.1f}/{total_gb:.1f} GB ({memory_percent:.0f}%)')
        self.view.label_runtime_info.setToolTip(snapshot.name)

    def handle_gpu_monitor_unavailable(self, _: str):
        self._gpu_monitor_available = False
        self.apply_gpu_memory_fallback()

    def apply_gpu_memory_fallback(self):
        gpu_stats = self.sample_gpu_memory_stats()
        if gpu_stats is None:
            self.view.progress_device_usage.hide()
            self.view.label_device_title.setText("CPU")
            self.view.label_device_percent.setText("")
            self.view.label_runtime_info.setText("Device: CPU training")
            return
        self.view.progress_device_usage.show()
        self.view.progress_device_usage.setValue(int(round(max(0, min(gpu_stats["memory_percent"], 100)))))
        self.view.label_device_title.setText("GPU Mem")
        self.view.label_device_percent.setText(f'{gpu_stats["memory_percent"]:.0f}%')
        self.view.label_runtime_info.setText(gpu_stats["text"])
        self.view.label_runtime_info.setToolTip(gpu_stats["device_name"])

    def update_epoch(self, current_epoch: int, current_ckpt_path: str):
        self.view.progress_epoch.setValue(current_epoch)
        self._current_epoch_for_eta = current_epoch + 1
        self.view.progress_iter.setValue(0)
        self.view.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self._worker.save_records_after_epoch(current_epoch, current_ckpt_path)
        self.refresh_training_time(current_epoch)
        self.view._refresh_training_chart(self._worker.get_train_summary(), self._worker.get_train_result_summary(), self._worker.get_valid_result_summary())
        self.view._refresh_training_history_table(self._worker.get_train_summary(), self._worker.get_train_result_summary(), self._worker.get_valid_result_summary())
        self.view._refresh_validation_table(current_epoch, self._worker.get_valid_result_summary())

    def update_iter(self, phase_type: str, iter_idx: int, iter_len: int):
        if phase_type == PHASE_TYPE_TRAINING:
            self.view.progress_iter.setStyleSheet(self.view.COLOR_TRAINING)
        elif phase_type == PHASE_TYPE_VALIDATION:
            self.view.progress_iter.setStyleSheet(self.view.COLOR_VALIDATION)
        else:
            raise NotImplementedError
        self.view.progress_iter.setMaximum(iter_len)
        self.view.progress_iter.setValue(iter_idx + 1)
        now = time.monotonic()
        is_last_iter = iter_len > 0 and (iter_idx + 1) >= iter_len
        if is_last_iter or (now - self._last_iter_ui_update_ts) >= 0.15:
            self._last_iter_ui_update_ts = now
            self.refresh_training_time_live(phase_type, iter_idx, iter_len)

    def close(self):
        self.stop_training("window close")

