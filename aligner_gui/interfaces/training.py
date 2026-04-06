"""
Training-layer interfaces (Dependency Inversion for TrainerViewModel).

Both ``ITrainerSession`` and ``ITrainingThread`` are structural protocols
(PEP 544).  ``ProjectSession`` and ``ThreadTrain`` satisfy them implicitly
— no inheritance needed.  Tests supply lightweight mocks instead.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ITrainerSession(Protocol):
    """Minimal session API consumed by :class:`~...trainer_viewmodel.TrainerViewModel`.

    ``ProjectSession`` satisfies this protocol.  During unit tests, inject
    :class:`~aligner_gui.interfaces.testing.MockTrainerSession` instead so
    no file-system or ML-backend initialisation is required.
    """

    # ------------------------------------------------------------------
    # Properties (read-only model state)
    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str:
        """Name of the primary validation metric (e.g. ``"mAP"``).  Read-only."""
        ...

    @property
    def project_path(self) -> str:
        """Absolute path to the project root directory."""
        ...

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_project_settings(self):
        """Return the current :class:`ProjectSettings` dataclass."""
        ...

    def set_project_settings(self, settings) -> None:
        """Persist *settings* to the project config file."""
        ...

    # ------------------------------------------------------------------
    # Model profile catalogue
    # ------------------------------------------------------------------

    def get_model_profiles(self):
        """Return the list of available detector profile descriptors."""
        ...

    # ------------------------------------------------------------------
    # Training summaries
    # ------------------------------------------------------------------

    def get_train_summary(self):
        """Return the live :class:`TrainSummary` (loss by epoch, etc.)."""
        ...

    def get_train_result_summary(self):
        """Return per-epoch train-split metric results."""
        ...

    def get_valid_result_summary(self):
        """Return per-epoch validation-split metric results."""
        ...

    def save_records_after_epoch(self, epoch: int, ckpt_path: str) -> None:
        """Persist training records produced at the end of *epoch*."""
        ...

    # ------------------------------------------------------------------
    # Resume state
    # ------------------------------------------------------------------

    def can_resume_training(self) -> bool:
        """Return ``True`` if a ``last.pth`` checkpoint is available."""
        ...

    def get_last_completed_epoch(self) -> int:
        """Return the epoch index of the most recently saved checkpoint (0 if none)."""
        ...

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def get_dataset_summary_path(self) -> str:
        """Return the path to the dataset summary JSON file."""
        ...


@runtime_checkable
class ITrainingThread(Protocol):
    """Minimal thread API consumed by :class:`~...trainer_viewmodel.TrainerViewModel`.

    ``ThreadTrain`` (a ``QThread`` subclass) satisfies this protocol.
    Tests use :class:`~aligner_gui.interfaces.testing.MockTrainingThread`.

    **Qt signals** cannot be declared in a ``Protocol``; they are documented
    here as comments.  Any concrete class must expose:

    .. code-block:: python

        qt_signal_stop_training : pyqtSignal(str)
            Emitted when the thread finishes.  Payload: stop reason string.
        qt_signal_update_epoch  : pyqtSignal(int, str)
            Emitted at the end of each epoch.  Payload: (epoch, ckpt_path).
        qt_signal_update_iter   : pyqtSignal(str, int, int)
            Emitted for each batch.  Payload: (phase, iter_idx, iter_len).
        qt_signal_status        : pyqtSignal(str)
            Emitted for informational status messages.
    """

    # Signal placeholders — ``Any`` keeps type-checkers happy while
    # acknowledging that .connect() / .emit() are the only operations used.
    qt_signal_stop_training: Any   # pyqtSignal(str)
    qt_signal_update_epoch: Any    # pyqtSignal(int, str)
    qt_signal_update_iter: Any     # pyqtSignal(str, int, int)
    qt_signal_status: Any          # pyqtSignal(str)

    def start(self) -> None:
        """Start the background thread."""
        ...

    def terminate(self) -> None:
        """Request the thread to stop immediately."""
        ...

    def isRunning(self) -> bool:
        """Return ``True`` while the thread's ``run()`` is executing."""
        ...

    def set_resume_training(self, resume: bool) -> None:
        """Configure whether this run should resume from the last checkpoint."""
        ...
