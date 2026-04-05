from __future__ import annotations

from PyQt5.QtCore import *

import logging
import traceback
from aligner_gui.utils import const

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class ThreadTrain(QThread):

    qt_signal_stop_training = pyqtSignal(str)
    qt_signal_update_epoch = pyqtSignal(int, str)
    qt_signal_update_iter = pyqtSignal(str, int, int)
    qt_signal_status = pyqtSignal(str)

    def __init__(self, worker):
        super().__init__()
        self._worker = worker
        self._resume_training = False

    def set_resume_training(self, enabled: bool):
        self._resume_training = bool(enabled)


    def _success(self):
        self._worker.success_training()
        self.qt_signal_stop_training.emit(const.SUCCESS)

    def _error(self):
        self._worker.close_logger()
        self.qt_signal_stop_training.emit(const.ERROR)

    def terminate(self):
        TRAIN_LOGGER.info("thread is terminated")
        self._worker.stop_training()
        self.wait(msecs=15000)
        super().terminate()

    def run(self):
        try:
            # setup_default_os_environ()
            self._main()
            self._success()
        except Exception as e:
            if str(e) != "Training quit cause of button click interrupt":
                traceback.print_tb(e.__traceback__)
                error_msg_detail = "ERROR - " + str(traceback.format_exc())
                TRAIN_LOGGER.error(error_msg_detail)
                error_msg = "ERROR - " + str(e)
                TRAIN_LOGGER.error(error_msg)
                self._error()


    def callback_one_iter_finished(self, phase_type: str, iter_idx: int, iter_len: int):
        self.qt_signal_update_iter.emit(phase_type, iter_idx, iter_len)

    def callback_one_epoch_finished(self, epoch, last_ckpt_path):
        self.qt_signal_update_epoch.emit(epoch, last_ckpt_path)

    def callback_status(self, message: str):
        TRAIN_LOGGER.info(message)
        self.qt_signal_status.emit(message)

    def _main(self):
        TRAIN_LOGGER.info("Preparing training...")
        self._worker.start_train(
            self.callback_one_epoch_finished,
            self.callback_one_iter_finished,
            self.callback_status,
            resume=self._resume_training,
        )
