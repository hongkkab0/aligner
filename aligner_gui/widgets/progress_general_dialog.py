from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from qtpy.QtWidgets import QDialog
from aligner_gui.ui.progress_dlg import Ui_progress_dlg
from typing import Callable, List, Tuple, Dict
import logging
import traceback


class ProgressGeneralThread(QThread):
    sig_processing = pyqtSignal(int, str) # index, message
    sig_completed = pyqtSignal(bool)

    def __init__(self, work):
        super().__init__()
        self.do_progress = True
        self.work = work

    def run(self) -> None:
        try:
            res = self.work(self.sig_processing)
        except Exception:
            logging.error("Background task failed.\n%s", traceback.format_exc())
            res = False
        self.sig_completed.emit(res)


class ProgressGeneralDialog(QDialog, Ui_progress_dlg):
    def __init__(self, title, work: Callable, jobs_len):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle(title)
        self.is_success = False
        self.is_manual_stop = True
        self.work = work
        self.setFixedSize(self.size())
        self.lbl_info.setText("")
        self.progress_bar.setMaximum(jobs_len)
        self._th = ProgressGeneralThread(work)
        self._th.sig_processing.connect(self.processing)
        self._th.sig_completed.connect(self.completed)
        self._th.start()



    def get_data(self):
        return self._th.data

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self._th.do_progress = False

    def processing(self, job_index: int, msg: str):
        self.progress_bar.setValue(job_index)
        self.lbl_info.setText(msg)

    def completed(self, res: bool):
        self.is_success = res
        self.is_manual_stop = False
        self.close()
