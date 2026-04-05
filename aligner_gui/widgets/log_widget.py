from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from aligner_gui.ui.log_widget import Ui_log_widget
from aligner_gui.utils.log_manager import Logger
import logging

class LogWidget(QWidget, Ui_log_widget):

    def __init__(self, prefixes=None):
        super().__init__()
        self.setupUi(self)
        self.logger = Logger(self.plain_text_edit_log, prefixes=prefixes)
        logging.getLogger().addHandler(self.logger)

    def close(self) -> bool:
        logging.getLogger().removeHandler(self.logger)
        return super().close()
