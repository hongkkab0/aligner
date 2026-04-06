"""
Testing-layer interfaces and mock stubs.

Exports
-------
ITesterSession   – Protocol for the session API used by TesterViewModel.
ITestingThread   – Protocol for the testing worker thread.
MockTrainerSession  – Lightweight test double for ITrainerSession.
MockTrainingThread  – Lightweight test double for ITrainingThread.
MockTesterSession   – Lightweight test double for ITesterSession.
MockTestingThread   – Lightweight test double for ITestingThread.

Example usage in a pytest test::

    from aligner_gui.interfaces import MockTrainingThread, MockTrainerSession
    from aligner_gui.viewmodels.trainer_viewmodel import TrainerViewModel

    def test_stop_training_terminates_thread(qtbot):
        session = MockTrainerSession()
        thread  = MockTrainingThread()
        vm = TrainerViewModel(session, training_thread=thread)
        vm.launch_training(resume=False)
        assert thread.isRunning()
        vm.stop_training("abort")
        assert not thread.isRunning()
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


# ===========================================================================
# ITesterSession
# ===========================================================================

@runtime_checkable
class ITesterSession(Protocol):
    """Minimal session API consumed by :class:`~...tester_viewmodel.TesterViewModel`.

    ``ProjectSession`` satisfies this protocol implicitly.  Tests inject
    :class:`MockTesterSession` so no file-system access is required.
    """

    @property
    def mean_test_time(self) -> float:
        """Average inference time per image in seconds."""
        ...

    def get_project_settings(self):
        """Return current :class:`ProjectSettings`."""
        ...

    def get_dataset_summary_path(self) -> str:
        """Return path to the dataset summary JSON file."""
        ...

    def is_there_trained_checkpoint(self) -> bool:
        """Return ``True`` if a trained model checkpoint is available."""
        ...

    def get_test_result_summary(self):
        """Return the most recent test result summary object."""
        ...


# ===========================================================================
# ITestingThread
# ===========================================================================

@runtime_checkable
class ITestingThread(Protocol):
    """Minimal thread API consumed by :class:`~...tester_viewmodel.TesterViewModel`.

    ``ThreadTest`` satisfies this protocol.  Tests use :class:`MockTestingThread`.

    **Qt signals** (documented; not declarable in a Protocol):

    .. code-block:: python

        qt_signal_stop_testing                  : pyqtSignal(str)
        qt_signal_update_iter                   : pyqtSignal(int, int)
        qt_signal_update_test_result_summary    : pyqtSignal()
    """

    qt_signal_stop_testing: Any                  # pyqtSignal(str)
    qt_signal_update_iter: Any                   # pyqtSignal(int, int)
    qt_signal_update_test_result_summary: Any    # pyqtSignal()

    def start(self) -> None: ...
    def terminate(self) -> None: ...
    def isRunning(self) -> bool: ...
    def set_img_paths_to_test(self, paths: list[str]) -> None: ...


# ===========================================================================
# Mock signal helper
# ===========================================================================

class _MockSignal:
    """Minimal signal shim that behaves like a ``pyqtSignal`` for testing.

    Supports ``.connect(callback)`` and ``.emit(*args)``.
    Does **not** require a QApplication or event loop.
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable] = []

    def connect(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable | None = None) -> None:
        if callback is None:
            self._callbacks.clear()
        else:
            self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    def emit(self, *args) -> None:
        for cb in list(self._callbacks):
            cb(*args)


# ===========================================================================
# MockTrainerSession
# ===========================================================================

class MockTrainerSession:
    """Test double that satisfies :class:`ITrainerSession`.

    All method return values are configurable attributes so tests can arrange
    the exact state needed without touching files or ML backends.
    """

    def __init__(self) -> None:
        self._metric_name: str = "mAP"
        self._project_path: str = "/mock/project"
        self._project_settings = None
        self._model_profiles: list = []
        self._can_resume: bool = False
        self._last_epoch: int = 0

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def project_path(self) -> str:
        return self._project_path

    def get_project_settings(self):
        return self._project_settings

    def set_project_settings(self, settings) -> None:
        self._project_settings = settings

    def get_model_profiles(self) -> list:
        return self._model_profiles

    def get_train_summary(self):
        return None

    def get_train_result_summary(self):
        return None

    def get_valid_result_summary(self):
        return None

    def save_records_after_epoch(self, epoch: int, ckpt_path: str) -> None:
        self._last_epoch = epoch

    def can_resume_training(self) -> bool:
        return self._can_resume

    def get_last_completed_epoch(self) -> int:
        return self._last_epoch

    def get_dataset_summary_path(self) -> str:
        return "/mock/dataset_summary.json"


# ===========================================================================
# MockTrainingThread
# ===========================================================================

class MockTrainingThread:
    """Test double that satisfies :class:`ITrainingThread`.

    Signals are replaced with :class:`_MockSignal` instances so
    ``.connect()`` / ``.emit()`` work without a Qt event loop.
    Thread lifecycle (``start`` / ``terminate`` / ``isRunning``) is
    tracked by a simple boolean flag.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._resume: bool = False

        self.qt_signal_stop_training = _MockSignal()
        self.qt_signal_update_epoch  = _MockSignal()
        self.qt_signal_update_iter   = _MockSignal()
        self.qt_signal_status        = _MockSignal()

    def start(self) -> None:
        self._running = True

    def terminate(self) -> None:
        self._running = False

    def isRunning(self) -> bool:
        return self._running

    def set_resume_training(self, resume: bool) -> None:
        self._resume = resume

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def simulate_epoch(self, epoch: int, ckpt_path: str = "") -> None:
        """Drive the epoch-updated signal for in-test simulation."""
        self.qt_signal_update_epoch.emit(epoch, ckpt_path)

    def simulate_iter(self, phase: str, idx: int, total: int) -> None:
        """Drive the iter-updated signal for in-test simulation."""
        self.qt_signal_update_iter.emit(phase, idx, total)

    def simulate_finish(self, reason: str = "success") -> None:
        """Drive the stop-training signal and mark thread as stopped."""
        self._running = False
        self.qt_signal_stop_training.emit(reason)


# ===========================================================================
# MockTesterSession
# ===========================================================================

class MockTesterSession:
    """Test double that satisfies :class:`ITesterSession`."""

    def __init__(self) -> None:
        self._mean_test_time: float = 0.0
        self._has_checkpoint: bool = False
        self._test_result_summary = None

    @property
    def mean_test_time(self) -> float:
        return self._mean_test_time

    def get_project_settings(self):
        return None

    def get_dataset_summary_path(self) -> str:
        return "/mock/dataset_summary.json"

    def is_there_trained_checkpoint(self) -> bool:
        return self._has_checkpoint

    def get_test_result_summary(self):
        return self._test_result_summary


# ===========================================================================
# MockTestingThread
# ===========================================================================

class MockTestingThread:
    """Test double that satisfies :class:`ITestingThread`."""

    def __init__(self) -> None:
        self._running: bool = False
        self._paths: list[str] = []

        self.qt_signal_stop_testing               = _MockSignal()
        self.qt_signal_update_iter                = _MockSignal()
        self.qt_signal_update_test_result_summary = _MockSignal()

    def start(self) -> None:
        self._running = True

    def terminate(self) -> None:
        self._running = False

    def isRunning(self) -> bool:
        return self._running

    def set_img_paths_to_test(self, paths: list[str]) -> None:
        self._paths = list(paths)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def simulate_iter(self, idx: int, total: int) -> None:
        self.qt_signal_update_iter.emit(idx, total)

    def simulate_finish(self, reason: str = "success") -> None:
        self._running = False
        self.qt_signal_stop_testing.emit(reason)
