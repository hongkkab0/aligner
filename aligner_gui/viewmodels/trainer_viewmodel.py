from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING, Callable, Optional, TypedDict

from PyQt5.QtCore import QTimer, pyqtSignal

from aligner_gui.trainer.gpu_monitor import GpuUsageSnapshot, NvidiaSmiPoller
from aligner_gui.trainer.thread_train import ThreadTrain
from aligner_gui.trainer.training_timer import TrainingTimer, timestamp2time
from aligner_gui.shared import const
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase
from aligner_engine.const import PHASE_TYPE_TRAINING, PHASE_TYPE_VALIDATION
from aligner_engine.project_settings import ProjectSettings

if TYPE_CHECKING:
    from aligner_gui.interfaces.training import ITrainerSession, ITrainingThread

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class DeviceUsageInfo(TypedDict):
    """Payload emitted by :attr:`TrainerViewModel.device_usage_updated`.

    visible  : Whether a GPU is available; False means CPU-only mode.
    title    : Label for the device panel header (e.g. "GPU Mem" or "CPU").
    percent  : Human-readable percentage string (e.g. "72%"), empty when invisible.
    value    : Integer 0-100 for the progress bar; 0 when invisible.
    info     : Detailed memory string shown in the panel body.
    tooltip  : Device name for the tooltip, empty when invisible.
    """
    visible: bool
    title: str
    percent: str
    value: int
    info: str
    tooltip: str


class TrainerViewModel(ViewModelBase):
    """Presentation-logic layer for the Trainer tab.

    Communication contract
    ----------------------
    View  → ViewModel : method calls only (commands).
    ViewModel → View  : pyqtSignal only (no ``self.view.*`` access).

    The View is responsible for all widget reads (e.g. checkbox states) and
    must pass those values as arguments when calling command methods.
    """

    # ------------------------------------------------------------------
    # Signals (ViewModel → View)
    # ------------------------------------------------------------------

    # Training lifecycle
    training_started = pyqtSignal(bool, int)    # (is_resume, start_epoch)
    training_stopped = pyqtSignal(str)           # reason constant

    # Per-epoch / per-iter progress
    epoch_updated = pyqtSignal(int, str)         # (epoch, ckpt_path)
    iter_updated = pyqtSignal(str, int, int)     # (phase_type, iter_idx, iter_len)
    training_time_updated = pyqtSignal(str)      # formatted time string
    status_label_updated = pyqtSignal(str)       # transient status message for label

    # Resume checkbox
    resume_state_changed = pyqtSignal(bool, int)  # (can_resume, last_epoch)

    # GPU / device panel
    device_usage_updated = pyqtSignal(dict)      # payload schema: DeviceUsageInfo

    # Global app status (consumed by MainWindowViewModel)
    app_status_changed = pyqtSignal(bool)        # True = training, False = idle

    def __init__(
        self,
        session: "ITrainerSession",
        tester_reload_callback: Optional[Callable[[], None]] = None,
        *,
        training_thread: "ITrainingThread | None" = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._tester_reload_callback = tester_reload_callback

        # Dependency injection: accept an externally-supplied thread (e.g. a
        # test double) or fall back to the real ThreadTrain.
        self._th_train: "ITrainingThread" = (
            training_thread if training_thread is not None else ThreadTrain(self._session)
        )
        self._th_train.qt_signal_stop_training.connect(self._on_thread_stopped)
        self._th_train.qt_signal_update_epoch.connect(self._on_thread_epoch_updated)
        self._th_train.qt_signal_update_iter.connect(self._on_thread_iter_updated)
        self._th_train.qt_signal_status.connect(self._on_thread_status)

        self._training_timer = TrainingTimer()
        self._device_poll_timer = QTimer(self)
        self._device_poll_timer.setInterval(1500)
        self._device_poll_timer.timeout.connect(self._poll_device_usage)

        self._gpu_monitor = NvidiaSmiPoller(self)
        self._gpu_monitor.stats_updated.connect(self._on_gpu_snapshot)
        self._gpu_monitor.unavailable.connect(self._on_gpu_unavailable)
        self._gpu_monitor_available: bool | None = None

        self._current_epoch_for_eta = 1
        self._last_iter_ui_update_ts = 0.0

        # Log startup environment paths for diagnostics
        TRAIN_LOGGER.info("=== DICE Aligner startup ===")
        TRAIN_LOGGER.info("Python: %s", sys.executable)
        for _key in ("dice_aligner_path", "dice_aligner_embed_python_path", "dice_aligner_python_path"):
            TRAIN_LOGGER.info("ENV %s = %s", _key, os.environ.get(_key, "(not set)"))

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_training(self) -> bool:
        return self._th_train.isRunning()

    @property
    def metric_name(self) -> str:
        return self._session.metric_name

    # ------------------------------------------------------------------
    # Settings (Model access abstracted away from View)
    # ------------------------------------------------------------------

    def get_settings(self) -> ProjectSettings:
        return self._session.get_project_settings()

    def save_settings(self, settings: ProjectSettings) -> None:
        self._session.set_project_settings(settings)

    def update_setting(self, **kwargs) -> None:
        settings = self._session.get_project_settings()
        for key, value in kwargs.items():
            setattr(settings, key, value)
        self._session.set_project_settings(settings)

    def get_model_profiles(self):
        return self._session.get_model_profiles()

    # ------------------------------------------------------------------
    # Training history (Model access abstracted away from View)
    # ------------------------------------------------------------------

    def get_train_summary(self):
        return self._session.get_train_summary()

    def get_train_result_summary(self):
        return self._session.get_train_result_summary()

    def get_valid_result_summary(self):
        return self._session.get_valid_result_summary()

    def save_records_after_epoch(self, epoch: int, ckpt_path: str) -> None:
        self._session.save_records_after_epoch(epoch, ckpt_path)

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def refresh_resume_state(self) -> None:
        can_resume = self._session.can_resume_training()
        last_epoch = self._session.get_last_completed_epoch()
        self.resume_state_changed.emit(can_resume, last_epoch)

    # ------------------------------------------------------------------
    # Training assets (work function for ProgressGeneralDialog)
    # ------------------------------------------------------------------

    def prepare_training_assets(self, processing_signal) -> bool:
        """Passed as the ``work`` arg of ProgressGeneralDialog.

        Called from the dialog's worker thread; raises on failure.
        """
        from aligner_gui.project.project_dataset_service import build_dataset_summary_from_project

        settings = self.get_settings()
        processing_signal.emit(1, "Building dataset summary...")
        build_dataset_summary_from_project(
            self._session.project_path,
            self._session.get_dataset_summary_path(),
            settings.include_empty,
        )
        processing_signal.emit(2, "Preparing training workspace...")
        return True

    # ------------------------------------------------------------------
    # Training lifecycle commands (called by View)
    # ------------------------------------------------------------------

    def validate_start_training(self, resume: bool) -> tuple[bool, str]:
        """Returns ``(ok, error_key)``."""
        if resume and not self._session.can_resume_training():
            return False, "no_checkpoint"
        if resume and self._session.get_last_completed_epoch() >= self.get_settings().max_epochs:
            return False, "max_epochs_reached"
        return True, ""

    def begin_training_prep(self, resume: bool) -> None:
        """Emit training_started so View updates UI before the prep dialog appears."""
        last_epoch = self._session.get_last_completed_epoch() if resume else 0
        settings = self.get_settings()
        self._training_timer.train_start(last_epoch, settings.max_epochs)
        self._current_epoch_for_eta = max(last_epoch + 1, 1)
        self._last_iter_ui_update_ts = 0.0
        self.training_started.emit(resume, last_epoch)

    def launch_training(self, resume: bool) -> None:
        """Start the background training thread (call after prep dialog succeeds)."""
        try:
            TRAIN_LOGGER.info("start training")
            if self._tester_reload_callback is not None:
                self._tester_reload_callback()
            self.app_status_changed.emit(True)
            self._gpu_monitor_available = None

            torch = self._get_torch()
            if torch.cuda.is_available():
                self._gpu_monitor.set_preferred_device_index(torch.cuda.current_device())

            self._device_poll_timer.start()
            self._poll_device_usage()

            self._th_train.set_resume_training(resume)
            self._th_train.start()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            TRAIN_LOGGER.error("ERROR - %s", e)
            self.app_status_changed.emit(False)
            self.training_stopped.emit(const.ERROR)

    def abort_training_prep(self) -> None:
        """Called by View when the preparation dialog fails or is cancelled."""
        self.training_stopped.emit(const.ERROR)

    def stop_training(self, reason: str) -> None:
        if self._th_train.isRunning():
            self._th_train.terminate()
        else:
            self._finalize_stop(reason)

    def close(self) -> None:
        if self._th_train.isRunning():
            self._th_train.terminate()

    # ------------------------------------------------------------------
    # ThreadTrain slots (private — forward to View via signals)
    # ------------------------------------------------------------------

    def _on_thread_stopped(self, reason: str) -> None:
        self._finalize_stop(reason)

    def _on_thread_epoch_updated(self, epoch: int, ckpt_path: str) -> None:
        self._current_epoch_for_eta = epoch + 1
        self.save_records_after_epoch(epoch, ckpt_path)
        self.epoch_updated.emit(epoch, ckpt_path)
        self._emit_training_time_epoch(epoch)

    def _on_thread_iter_updated(self, phase: str, idx: int, total: int) -> None:
        self.iter_updated.emit(phase, idx, total)
        now = time.monotonic()
        is_last = total > 0 and (idx + 1) >= total
        if is_last or (now - self._last_iter_ui_update_ts) >= 0.15:
            self._last_iter_ui_update_ts = now
            self._emit_training_time_live(phase, idx, total)

    def _on_thread_status(self, message: str) -> None:
        self.status_label_updated.emit(message)

    def _finalize_stop(self, reason: str) -> None:
        self._device_poll_timer.stop()
        self._gpu_monitor.stop()
        self._gpu_monitor_available = None

        torch = self._get_torch()
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

        self.app_status_changed.emit(False)
        self.training_stopped.emit(reason)
        self.refresh_resume_state()

    # ------------------------------------------------------------------
    # Time-label helpers
    # ------------------------------------------------------------------

    def _emit_training_time_epoch(self, cur_epoch: int) -> None:
        _one, avg, processed, remaining = self._training_timer.one_epoch_done(cur_epoch)
        self.training_time_updated.emit(
            "Avg epoch %.1fs   Elapsed %s   Remaining %s"
            % (avg, timestamp2time(processed), timestamp2time(remaining))
        )

    def _emit_training_time_live(self, phase: str, idx: int, total: int) -> None:
        epoch = min(max(self._current_epoch_for_eta, 1), 99999)
        avg_step, processed, remaining = self._training_timer.one_iter_progress(
            phase, idx, total, epoch
        )
        phase_name = "Train" if phase == PHASE_TYPE_TRAINING else "Valid"
        self.training_time_updated.emit(
            f"{phase_name} {idx + 1}/{total}   Avg step {avg_step:.2f}s"
            f"   Elapsed {timestamp2time(processed)}   Remaining {timestamp2time(remaining)}"
        )

    # ------------------------------------------------------------------
    # GPU / device monitoring (results forwarded to View via signal)
    # ------------------------------------------------------------------

    def _poll_device_usage(self) -> None:
        torch = self._get_torch()
        if not torch.cuda.is_available():
            self.device_usage_updated.emit(
                {"visible": False, "title": "CPU", "percent": "", "info": "Device: CPU training", "tooltip": ""}
            )
            return
        if self._gpu_monitor_available is not True:
            self._emit_gpu_fallback()
        self._gpu_monitor.poll()

    def _on_gpu_snapshot(self, snapshot: GpuUsageSnapshot) -> None:
        self._gpu_monitor_available = True
        pct = max(0.0, min(snapshot.memory_percent, 100.0))
        self.device_usage_updated.emit({
            "visible": True,
            "title": "GPU Mem",
            "percent": f"{pct:.0f}%",
            "value": int(round(pct)),
            "info": f"{snapshot.memory_used_mb / 1024:.1f}/{snapshot.memory_total_mb / 1024:.1f} GB ({pct:.0f}%)",
            "tooltip": snapshot.name,
        })

    def _on_gpu_unavailable(self, _: str) -> None:
        self._gpu_monitor_available = False
        self._emit_gpu_fallback()

    def _emit_gpu_fallback(self) -> None:
        torch = self._get_torch()
        if not torch.cuda.is_available():
            self.device_usage_updated.emit(
                {"visible": False, "title": "CPU", "percent": "", "info": "Device: CPU training", "tooltip": ""}
            )
            return
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        try:
            free, total = torch.cuda.mem_get_info(device_index)
            total = max(total, 1)
            used = max(total - free, 0)
        except Exception:
            props = torch.cuda.get_device_properties(device_index)
            total = max(props.total_memory, 1)
            used = max(torch.cuda.memory_allocated(device_index),
                       torch.cuda.memory_reserved(device_index))
        pct = used * 100.0 / total
        self.device_usage_updated.emit({
            "visible": True,
            "title": "GPU Mem",
            "percent": f"{pct:.0f}%",
            "value": int(round(max(0, min(pct, 100)))),
            "info": f"{used / (1024**3):.1f}/{total / (1024**3):.1f} GB ({pct:.0f}%)",
            "tooltip": device_name,
        })

    @staticmethod
    def _get_torch():
        import torch
        return torch
