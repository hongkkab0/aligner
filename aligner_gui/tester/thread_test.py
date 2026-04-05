from __future__ import annotations

from PyQt5.QtCore import *

import logging
import traceback
from aligner_gui.utils import const

TEST_LOGGER = logging.getLogger("aligner.tester")

class ThreadTest(QThread):
    qt_signal_stop_testing = pyqtSignal(str)
    qt_signal_update_iter = pyqtSignal(int, int)
    qt_signal_update_test_result_summary = pyqtSignal()

    def __init__(self, worker):
        super().__init__()
        self._worker = worker
        self._img_paths_to_test = []

    def _success(self):
        self._worker.success_test()
        self.qt_signal_stop_testing.emit(const.SUCCESS)

    def _error(self):
        self.qt_signal_stop_testing.emit(const.ERROR)

    def terminate(self):
        TEST_LOGGER.info("thread is terminated")
        self._worker.stop_test()
        self.wait()
        super().terminate()

    def run(self):
        try:
            # setup_default_os_environ()
            self._main()
            self._success()
            self.qt_signal_update_test_result_summary.emit()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            TEST_LOGGER.error(error_msg)
            self._error()


    def callback_one_iter_finished(self, iter_idx: int, iter_len: int):
        self.qt_signal_update_iter.emit(iter_idx, iter_len)


    def set_img_paths_to_test(self, img_paths):
        self._img_paths_to_test = img_paths


    def _main(self):
        self._worker.start_test(self.callback_one_iter_finished, self._img_paths_to_test)
