from __future__ import annotations

import logging
import os
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING

from aligner_gui.tester.thread_test import ThreadTest
from aligner_gui.shared import const, gui_util
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase

if TYPE_CHECKING:
    from aligner_gui.tester.tester_view import TesterView

TEST_LOGGER = logging.getLogger("aligner.tester")


class TesterViewModel(ViewModelBase):
    def __init__(self, view: 'TesterView', session):
        super().__init__(view)
        self.view = view
        self._worker = session
        self._th_test = ThreadTest(self._worker)

    def initialize(self):
        self._th_test.qt_signal_stop_testing.connect(self.stop_testing)
        self._th_test.qt_signal_update_iter.connect(self.update_iter)
        self._th_test.qt_signal_update_test_result_summary.connect(self.update_test_result_summary)

    def get_torch(self):
        import torch
        return torch

    def append_files(self, paths):
        merged = list(self.view._file_list)
        seen = {path.lower(): path for path in merged}
        for path in paths:
            normalized = os.path.abspath(path)
            key = normalized.lower()
            if key not in seen:
                seen[key] = normalized
                merged.append(normalized)
        self.view._file_list = merged

    def update_iter(self, iter_idx: int, iter_len: int):
        self.view.progress_iter.setStyleSheet(self.view.COLOR_INFERENCE)
        self.view.progress_iter.setMaximum(iter_len)
        self.view.progress_iter.setValue(iter_idx + 1)
        self.view.btn_test.setEnabled(True)

    def handle_test_button_clicked(self):
        if self.view.btn_test.isChecked():
            self.start_test()
        else:
            self.stop_testing("manual stop")

    def start_test(self):
        try:
            self.emit_status("Starting test...")
            if not self._worker.is_there_trained_checkpoint():
                gui_util.get_message_box(self.view, "Invalid Test", "There is no trained model.")
                self.view.btn_test.setChecked(False)
                return
            TEST_LOGGER.info('Test started.')
            self._th_test.set_img_paths_to_test(deepcopy(self.view._file_list))
            self.on_start_test()
            self._th_test.start()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            TEST_LOGGER.error("ERROR - %s", e)
            self.stop_testing(const.ERROR)

    def stop_testing(self, reason: str):
        if self._th_test.isRunning():
            self._th_test.terminate()

        torch = self.get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reason == const.SUCCESS:
            TEST_LOGGER.info("test finished successfully")
            self.emit_status("Test finished successfully.", 3000)
        elif reason == const.ERROR:
            TEST_LOGGER.error("test failed")
            self.emit_status("Test failed.", 3000)
        else:
            TEST_LOGGER.info(reason)
            self.emit_status(reason, 3000)
        self.on_stop_test()

    def close(self):
        self.stop_testing("window close")
        self.view._preview_renderer.clear()

    def on_start_test(self):
        self.view.btn_test.setChecked(True)
        self.view.btn_test.setEnabled(False)
        self.view.progress_iter.setMaximum(1)
        self.view.progress_iter.setValue(0)
        self.view.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.view.lbl_test_indicator.show()

    def on_stop_test(self):
        self.view.btn_test.setChecked(False)
        self.view.btn_test.setEnabled(True)
        self.view.progress_iter.setMaximum(1)
        self.view.progress_iter.setValue(0)
        self.view.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.view.lbl_test_indicator.hide()

    def update_test_result_summary(self):
        self.view._worker_test_result_summary = self._worker.get_test_result_summary()
        self.view._clear_preview_cache()
        self.view._refresh_test_detail_table()
        self.view._refresh_test_time()

