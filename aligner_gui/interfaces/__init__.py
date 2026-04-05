"""
aligner_gui.interfaces
======================

Abstract interfaces (PEP 544 ``Protocol``) for the two major cross-layer
boundaries in Aligner's MVVM stack:

  * **Session facades** – describe the minimal API each ViewModel needs
    from the model layer (ProjectSession).  Concrete code satisfies these
    protocols implicitly via duck-typing; tests inject lightweight mocks.

  * **Worker threads** – describe the API each ViewModel needs from its
    background QThread.  Thread objects carry Qt signals, so the interface
    documents signal names and their payload types as comments (runtime
    signals cannot be declared in a ``Protocol``).  Tests use
    :class:`~aligner_gui.interfaces.testing.MockTrainingThread` /
    :class:`~aligner_gui.interfaces.testing.MockTestingThread`.

Usage example (unit test without any Qt infrastructure)::

    from aligner_gui.interfaces.testing import MockTrainingThread, MockTrainerSession
    from aligner_gui.viewmodels.trainer_viewmodel import TrainerViewModel

    def test_training_stops_on_abort():
        mock_session = MockTrainerSession()
        mock_thread  = MockTrainingThread()
        vm = TrainerViewModel(mock_session, training_thread=mock_thread)
        vm.launch_training(resume=False)
        assert mock_thread.isRunning()
        vm.stop_training("abort")
        assert not mock_thread.isRunning()
"""

from aligner_gui.interfaces.training import ITrainerSession, ITrainingThread
from aligner_gui.interfaces.testing import (
    ITesterSession,
    ITestingThread,
    MockTrainerSession,
    MockTrainingThread,
    MockTesterSession,
    MockTestingThread,
)

__all__ = [
    "ITrainerSession",
    "ITrainingThread",
    "ITesterSession",
    "ITestingThread",
    "MockTrainerSession",
    "MockTrainingThread",
    "MockTesterSession",
    "MockTestingThread",
]
