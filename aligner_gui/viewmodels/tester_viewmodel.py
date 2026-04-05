from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import List

from PyQt5.QtCore import QObject, pyqtSignal

from aligner_gui.utils import const
from aligner_gui.tester.thread_test import ThreadTest

TEST_LOGGER = logging.getLogger("aligner.tester")


class TesterViewModel(QObject):
    """Presentation-logic layer for the Tester tab.

    Responsibilities
    ----------------
    * Owns :class:`~aligner_gui.tester.thread_test.ThreadTest` and manages
      its lifecycle.
    * Owns the canonical file list and class metadata loaded from the dataset
      summary.
    * Validates pre-conditions for starting inference.
    * Emits signals that the View (TesterWidget) connects to for UI updates.

    The View must NOT hold a reference to the session or the engine Worker; it
    interacts exclusively through this ViewModel.
    """

    # -- Inference lifecycle
    testing_started = pyqtSignal()
    testing_stopped = pyqtSignal(str)          # reason string

    # -- Progress forwarded from ThreadTest
    iter_updated = pyqtSignal(int, int)        # (iter_idx, iter_len)

    # -- Emitted after the test thread finishes and result summary is ready
    results_updated = pyqtSignal()

    # -- Emitted whenever the file list changes (initial load, add, remove, reset)
    file_list_changed = pyqtSignal(list)

    def __init__(self, session, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self._file_list: List[str] = []
        self._classes: List[str] = []
        self._class_index: dict = {}
        self._class_name: dict = {}
        self._dataset_summary: dict = {}
        self._test_result_summary = None

        self._th_test = ThreadTest(session)
        self._th_test.qt_signal_stop_testing.connect(self._on_thread_stopped)
        self._th_test.qt_signal_update_iter.connect(self._on_thread_iter_updated)
        self._th_test.qt_signal_update_test_result_summary.connect(self._on_thread_results_ready)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_testing(self) -> bool:
        return self._th_test.isRunning()

    @property
    def file_list(self) -> List[str]:
        return list(self._file_list)

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def class_index(self) -> dict:
        return dict(self._class_index)

    @property
    def class_name(self) -> dict:
        return dict(self._class_name)

    @property
    def test_result_summary(self):
        return self._test_result_summary

    @property
    def mean_test_time(self) -> float:
        return self._session.mean_test_time

    # ------------------------------------------------------------------
    # Settings / session access
    # ------------------------------------------------------------------

    def get_project_settings(self):
        return self._session.get_project_settings()

    def has_trained_checkpoint(self) -> bool:
        return self._session.is_there_trained_checkpoint()

    def get_dataset_summary_path(self) -> str:
        return self._session.get_dataset_summary_path()

    # ------------------------------------------------------------------
    # File list management
    # ------------------------------------------------------------------

    def reload_file_list(self) -> None:
        """Reload the file list from the dataset summary JSON."""
        try:
            dataset_summary_path = self._session.get_dataset_summary_path()
            with open(dataset_summary_path, "r", encoding="utf-8") as f:
                self._dataset_summary = json.load(f)
        except Exception as e:
            TEST_LOGGER.info(e)
            return

        self._classes = [c["name"] for c in self._dataset_summary["class_summary"]["classes"]]
        self._class_index = {v: idx for idx, v in enumerate(self._classes)}
        self._class_name = {idx: v for idx, v in enumerate(self._classes)}
        self._file_list = []
        self._append_unique([data["img_path"] for data in self._dataset_summary["data_summary"]])
        self.file_list_changed.emit(list(self._file_list))

    def add_files(self, paths: List[str]) -> None:
        """Add image paths to the file list (deduplicates by normalised path)."""
        self._append_unique(paths)
        self.file_list_changed.emit(list(self._file_list))

    def remove_files_at_rows(self, rows: List[int]) -> None:
        """Remove files by their row index (highest-index first to preserve order)."""
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self._file_list):
                self._file_list.pop(row)
        self.file_list_changed.emit(list(self._file_list))

    def _append_unique(self, paths: List[str]) -> None:
        seen = {p.lower(): p for p in self._file_list}
        for path in paths:
            normalised = os.path.abspath(path)
            key = normalised.lower()
            if key not in seen:
                seen[key] = normalised
                self._file_list.append(normalised)

    # ------------------------------------------------------------------
    # Testing lifecycle
    # ------------------------------------------------------------------

    def start_testing(self) -> None:
        if self.is_testing:
            return
        if not self.has_trained_checkpoint():
            TEST_LOGGER.warning("No trained checkpoint available.")
            self.testing_stopped.emit("no_checkpoint")
            return
        try:
            TEST_LOGGER.info("Test started.")
            self._th_test.set_img_paths_to_test(deepcopy(self._file_list))
            self.testing_started.emit()
            self._th_test.start()
        except Exception as e:
            TEST_LOGGER.error("ERROR - %s", e)
            self.testing_stopped.emit(const.ERROR)

    def stop_testing(self, reason: str) -> None:
        if self._th_test.isRunning():
            self._th_test.terminate()
        else:
            self.testing_stopped.emit(reason)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._th_test.isRunning():
            self._th_test.terminate()

    # ------------------------------------------------------------------
    # Slots forwarded from ThreadTest
    # ------------------------------------------------------------------

    def _on_thread_stopped(self, reason: str) -> None:
        self.testing_stopped.emit(reason)

    def _on_thread_iter_updated(self, iter_idx: int, iter_len: int) -> None:
        self.iter_updated.emit(iter_idx, iter_len)

    def _on_thread_results_ready(self) -> None:
        self._test_result_summary = self._session.get_test_result_summary()
        self.results_updated.emit()
