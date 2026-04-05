from __future__ import annotations

import logging
import os

from PyQt5 import QtGui
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QPlainTextEdit


LOG_FORMAT = "%(asctime)-15s [%(levelname)s] %(name)s: %(message)s"


class PrefixFilter(logging.Filter):
    def __init__(self, prefixes=None):
        super().__init__()
        self._prefixes = tuple(prefixes or ())

    def filter(self, record):
        if not self._prefixes:
            return True
        return any(
            record.name == prefix or record.name.startswith(prefix + ".")
            for prefix in self._prefixes
        )


class Logger(QObject, logging.Handler):
    MAXIMUM_BLOCK_COUNT = 1000

    _updateLog = pyqtSignal(str)

    def __init__(self, widget: QPlainTextEdit, prefixes=None):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.widget = widget
        self.widget.document().setMaximumBlockCount(self.MAXIMUM_BLOCK_COUNT)
        self.setFormatter(logging.Formatter(LOG_FORMAT))
        if prefixes:
            self.addFilter(PrefixFilter(prefixes))
        self._updateLog.connect(self._updateLog_mainTh)

    def _updateLog_mainTh(self, msg):
        self.widget.appendPlainText(msg)
        self.widget.moveCursor(QtGui.QTextCursor.End)

    def emit(self, record):
        self._updateLog.emit(self.format(record))


def configure_project_logging(project_path: str):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)
    log_dir = os.path.join(project_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_aligner_project_handler", False):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    file_specs = [
        ("app.log", None),
        ("trainer.log", ("aligner.trainer",)),
        ("tester.log", ("aligner.tester",)),
    ]

    for filename, prefixes in file_specs:
        handler = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
        handler.setFormatter(formatter)
        handler._aligner_project_handler = True
        if prefixes:
            handler.addFilter(PrefixFilter(prefixes))
        root_logger.addHandler(handler)

    if not any(isinstance(handler, logging.StreamHandler) and not getattr(handler, "_aligner_project_handler", False)
               for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
