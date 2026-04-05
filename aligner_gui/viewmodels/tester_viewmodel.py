from __future__ import annotations

import logging
import os
import traceback

from PyQt5.QtCore import pyqtSignal

from aligner_gui.tester.thread_test import ThreadTest
from aligner_gui.shared import const
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase

TEST_LOGGER = logging.getLogger("aligner.tester")


class TesterViewModel(ViewModelBase):
    """Presentation-logic layer for the Tester tab.

    Communication contract
    ----------------------
    View  → ViewModel : method calls only (commands).
    ViewModel → View  : pyqtSignal only (no ``self.view.*`` access).

    The ViewModel owns ``_file_list`` as the single source of truth.
    The View reads it via :meth:`get_file_list` and mutates it via command methods.
    """

    # ------------------------------------------------------------------
    # Signals (ViewModel → View)
    # ------------------------------------------------------------------
    testing_started = pyqtSignal()
    testing_stopped = pyqtSignal(str)           # reason constant
    iter_progress_updated = pyqtSignal(int, int)  # (iter_idx, iter_len)
    results_updated = pyqtSignal()
    test_blocked = pyqtSignal(str, str)         # (title, message) — view shows a dialog

    def __init__(self, session, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._th_test = ThreadTest(self._session)
        self._file_list: list[str] = []
        self._test_result_summary = None

        self._th_test.qt_signal_stop_testing.connect(self._on_thread_stopped)
        self._th_test.qt_signal_update_iter.connect(self._on_thread_iter_updated)
        self._th_test.qt_signal_update_test_result_summary.connect(self._on_thread_results_ready)

    # ------------------------------------------------------------------
    # Read-only session access (Model abstracted from View)
    # ------------------------------------------------------------------

    def get_project_settings(self):
        return self._session.get_project_settings()

    def get_dataset_summary_path(self) -> str:
        return self._session.get_dataset_summary_path()

    def get_mean_test_time(self) -> float:
        return self._session.mean_test_time

    def get_test_result_summary(self):
        return self._test_result_summary

    # ------------------------------------------------------------------
    # File-list management (single source of truth)
    # ------------------------------------------------------------------

    def get_file_list(self) -> list[str]:
        return list(self._file_list)

    def append_files(self, paths: list[str]) -> None:
        seen = {p.lower() for p in self._file_list}
        for path in paths:
            normalized = os.path.abspath(path)
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                self._file_list.append(normalized)

    def remove_files_at_rows(self, rows: list[int]) -> None:
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self._file_list):
                self._file_list.pop(row)

    def reset_file_list(self, paths: list[str]) -> None:
        self._file_list = []
        self.append_files(paths)

    # ------------------------------------------------------------------
    # Test lifecycle commands (called by View)
    # ------------------------------------------------------------------

    def handle_test_button_clicked(self, is_checked: bool) -> None:
        if is_checked:
            self.start_test()
        else:
            self.stop_testing("manual stop")

    def start_test(self) -> None:
        try:
            self.emit_status("Starting test...")
            if not self._session.is_there_trained_checkpoint():
                self.test_blocked.emit("Invalid Test", "There is no trained model.")
                self.testing_stopped.emit(const.ERROR)
                return
            TEST_LOGGER.info("Test started.")
            self._th_test.set_img_paths_to_test(list(self._file_list))
            self.testing_started.emit()
            self._th_test.start()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            TEST_LOGGER.error("ERROR - %s", e)
            self.stop_testing(const.ERROR)

    def stop_testing(self, reason: str) -> None:
        if self._th_test.isRunning():
            self._th_test.terminate()

        torch = self._get_torch()
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
        self.testing_stopped.emit(reason)

    def close(self) -> None:
        self.stop_testing("window close")

    # ------------------------------------------------------------------
    # Thread slots (private)
    # ------------------------------------------------------------------

    def _on_thread_stopped(self, reason: str) -> None:
        self.stop_testing(reason)

    def _on_thread_iter_updated(self, idx: int, total: int) -> None:
        self.iter_progress_updated.emit(idx, total)

    def _on_thread_results_ready(self) -> None:
        self._test_result_summary = self._session.get_test_result_summary()
        self.results_updated.emit()

    @staticmethod
    def _get_torch():
        import torch
        return torch
