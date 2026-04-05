from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class ViewModelBase(QObject):
    status_message_requested = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_status(self, message: str, timeout_ms: int = 0):
        self.status_message_requested.emit(message, timeout_ms)
