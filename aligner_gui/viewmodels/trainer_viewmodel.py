from __future__ import annotations

import logging
import traceback
from typing import Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from aligner_gui.utils import const
from aligner_gui.trainer.thread_train import ThreadTrain
from aligner_engine.project_settings import ProjectSettings

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class TrainerViewModel(QObject):
    """Presentation-logic layer for the Trainer tab.

    Responsibilities
    ----------------
    * Owns :class:`~aligner_gui.trainer.thread_train.ThreadTrain` and manages
      its lifecycle.
    * Reads / writes :class:`~aligner_engine.project_settings.ProjectSettings`
      through the session (never touching the engine Worker directly from the
      View).
    * Validates pre-conditions for starting or resuming training.
    * Prepares training assets (dataset summary) so the View only needs to
      wrap the work function in a progress dialog.
    * Emits fine-grained signals that the View connects to for UI updates.

    The View (TrainerWidget) must NOT hold a reference to the session or the
    engine Worker; it interacts exclusively through this ViewModel.
    """

    # -- Emitted when the ViewModel has validated the request and is ready
    #    for the View to show the prep dialog and then call launch_training().
    training_prep_started = pyqtSignal(bool, int)  # (is_resume, start_epoch)

    # -- Emitted once the background thread is actually running.
    training_launched = pyqtSignal()

    # -- Emitted when training ends for any reason (SUCCESS / ERROR / user msg).
    training_stopped = pyqtSignal(str)

    # -- Progress updates forwarded from ThreadTrain.
    epoch_updated = pyqtSignal(int, str)       # (epoch, ckpt_path)
    iter_updated = pyqtSignal(str, int, int)   # (phase_type, iter_idx, iter_len)
    status_message_changed = pyqtSignal(str)

    # -- Emitted whenever the resume availability changes.
    resume_state_changed = pyqtSignal(bool, int)  # (can_resume, last_epoch)

    def __init__(
        self,
        session,
        app_viewmodel,
        tester_reload_callback: Optional[Callable[[], None]] = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._app_vm = app_viewmodel
        self._tester_reload_callback = tester_reload_callback

        self._th_train = ThreadTrain(session)
        self._th_train.qt_signal_stop_training.connect(self._on_thread_stopped)
        self._th_train.qt_signal_update_epoch.connect(self._on_thread_epoch_updated)
        self._th_train.qt_signal_update_iter.connect(self._on_thread_iter_updated)
        self._th_train.qt_signal_status.connect(self._on_thread_status)

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
    # Settings access
    # ------------------------------------------------------------------

    def get_settings(self) -> ProjectSettings:
        return self._session.get_project_settings()

    def update_setting(self, **kwargs) -> None:
        """Apply one or more field updates to ProjectSettings and persist."""
        settings = self._session.get_project_settings()
        for key, value in kwargs.items():
            setattr(settings, key, value)
        self._session.set_project_settings(settings)

    def get_model_profiles(self):
        return self._session.get_model_profiles()

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def can_resume(self) -> bool:
        return self._session.can_resume_training()

    def last_completed_epoch(self) -> int:
        return self._session.get_last_completed_epoch()

    def refresh_resume_state(self) -> None:
        """Emit the current resume availability so the View can sync its UI."""
        can_resume = self._session.can_resume_training()
        last_epoch = self._session.get_last_completed_epoch()
        self.resume_state_changed.emit(can_resume, last_epoch)

    # ------------------------------------------------------------------
    # Training history / summaries
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
    # Training lifecycle (called by the View)
    # ------------------------------------------------------------------

    def validate_start_training(self, resume: bool) -> tuple[bool, str]:
        """Check pre-conditions.  Returns ``(ok, error_key)``."""
        if resume and not self._session.can_resume_training():
            return False, "no_checkpoint"
        if resume and self._session.get_last_completed_epoch() >= self.get_settings().max_epochs:
            return False, "max_epochs_reached"
        return True, ""

    def begin_training_prep(self, resume: bool) -> None:
        """Signal the View to update its UI to the 'preparing' state.

        Call this after :meth:`validate_start_training` succeeds and before
        showing the preparation dialog.
        """
        last_epoch = self._session.get_last_completed_epoch() if resume else 0
        self.training_prep_started.emit(resume, last_epoch)

    def prepare_training_assets(self, processing_signal) -> bool:
        """Work function intended to be called inside a ProgressGeneralDialog.

        The dialog's worker thread invokes this, passing its own
        ``processing_signal`` so we can emit step progress.  Returns True on
        success; any exception propagates to the dialog.
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

    def launch_training(self, resume: bool) -> None:
        """Start the training thread.

        Call this only after :meth:`prepare_training_assets` has succeeded.
        Notifies AppViewModel so it can lock export/project actions.
        """
        try:
            TRAIN_LOGGER.info("start training")
            if self._tester_reload_callback is not None:
                self._tester_reload_callback()
            self._app_vm.set_training()
            self._th_train.set_resume_training(resume)
            self._th_train.start()
            self.training_launched.emit()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            TRAIN_LOGGER.error("ERROR - %s", e)
            self._app_vm.set_idle()
            self.training_stopped.emit(const.ERROR)

    def abort_training_prep(self) -> None:
        """Called by the View when the preparation step fails or is cancelled."""
        self.training_stopped.emit(const.ERROR)

    def stop_training(self, reason: str) -> None:
        """Request termination; if the thread is running it will emit stopped."""
        if self._th_train.isRunning():
            self._th_train.terminate()
        else:
            # Thread never started (e.g. prep failed) – emit directly.
            self.training_stopped.emit(reason)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._th_train.isRunning():
            self._th_train.terminate()

    # ------------------------------------------------------------------
    # Slots forwarded from ThreadTrain
    # ------------------------------------------------------------------

    def _on_thread_stopped(self, reason: str) -> None:
        self._app_vm.set_idle()
        self.training_stopped.emit(reason)

    def _on_thread_epoch_updated(self, epoch: int, ckpt_path: str) -> None:
        self.epoch_updated.emit(epoch, ckpt_path)

    def _on_thread_iter_updated(self, phase: str, idx: int, total: int) -> None:
        self.iter_updated.emit(phase, idx, total)

    def _on_thread_status(self, message: str) -> None:
        self.status_message_changed.emit(message)
