from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal

from aligner_gui.utils import const


class AppViewModel(QObject):
    """Manages global application state shared across all tabs.

    The MainWindow owns this object and connects its ``status_changed`` signal
    to enable/disable toolbar actions.  TrainerViewModel notifies this object
    when training starts or finishes so that the rest of the UI can reflect the
    busy state without the Trainer widget needing a direct reference to
    MainWindow.
    """

    status_changed = pyqtSignal(str)  # emits the new APP_STATUS_* constant

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status: str = const.APP_STATUS_IDLE

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def app_status(self) -> str:
        return self._status

    @property
    def is_idle(self) -> bool:
        return self._status == const.APP_STATUS_IDLE

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_idle(self) -> None:
        if self._status == const.APP_STATUS_IDLE:
            return
        self._status = const.APP_STATUS_IDLE
        self.status_changed.emit(self._status)

    def set_training(self) -> None:
        if self._status == const.APP_STATUS_TRAINING:
            return
        self._status = const.APP_STATUS_TRAINING
        self.status_changed.emit(self._status)
