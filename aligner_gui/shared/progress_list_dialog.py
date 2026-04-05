import os.path as osp

from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from qtpy.QtWidgets import QDialog
from aligner_gui.ui.progress_dlg import Ui_progress_dlg
from typing import Callable, List, Tuple, Dict


class ProgressListThread(QThread):
    sig_processing = pyqtSignal(int)
    sig_completed = pyqtSignal()

    def __init__(self, work, length):
        super().__init__()
        self.do_progress = True
        self.work = work
        self.length = length

    def run(self) -> None:
        if self.length > 0:
            for idx in range(self.length):
                if self.do_progress:
                    self.work(idx)
                    self.sig_processing.emit(idx + 1)
        self.sig_completed.emit()


class ProgressListDialog(QDialog, Ui_progress_dlg):
    def __init__(self, work: Callable, jobs_to_do: List):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Progressing...")
        self.work = work
        self._jobs_to_do = jobs_to_do
        self.setFixedSize(self.size())
        self.lbl_info.setText("")
        self.progress_bar.setMaximum(len(jobs_to_do))
        self._th = ProgressListThread(work, len(jobs_to_do))
        self._th.sig_processing.connect(self.processing)
        self._th.sig_completed.connect(self.completed)
        self._th.start()

    def get_data(self):
        return self._th.data

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self._th.do_progress = False

    def processing(self, i: int):
        self.progress_bar.setValue(i)
        info = "Processing... - " + str(self._jobs_to_do[i - 1])
        self.lbl_info.setText(info)

    def completed(self):
        self.close()



