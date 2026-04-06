from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt5.QtCore import QTimer, pyqtSignal

from aligner_gui.labeler.file_list_service import remove_paths_from_file_list
from aligner_gui.project.project_dataset_service import load_labeler_image_list, save_labeler_image_list
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase

if TYPE_CHECKING:
    from aligner_gui.labeler.labeler_view import ImageIndexThread
    from aligner_gui.labeler.libs.shape import Shape

logger = logging.getLogger(__name__)


class LabelerViewModel(ViewModelBase):
    """Presentation-logic layer for the Labeler tab.

    Communication contract
    ----------------------
    View  → ViewModel : method calls only (commands).
    ViewModel → View  : pyqtSignal only (no ``self.view.*`` access).

    The ViewModel owns all file-list state as the single source of truth.
    The View reads state via query methods and reacts to signals.
    """

    # ------------------------------------------------------------------
    # Signals: image indexing lifecycle
    # ------------------------------------------------------------------
    image_index_started = pyqtSignal(int)            # total image count
    image_index_progress = pyqtSignal(int, int, str) # (cur, total, path)
    image_index_completed = pyqtSignal(bool)         # was_cancelled
    image_index_failed = pyqtSignal(str)             # error message

    # ------------------------------------------------------------------
    # Signals: file list state changes
    # ------------------------------------------------------------------
    file_list_cleared = pyqtSignal()
    file_list_batch_ready = pyqtSignal(object)       # list[(path, has_label, is_empty, needs_confirm)]
    file_list_enabled_changed = pyqtSignal(bool)
    file_list_item_state_changed = pyqtSignal(str, bool, bool, bool)  # (path, has_label, is_empty, needs_confirm)
    file_list_items_removed = pyqtSignal(object)     # set[str] of removed paths
    file_list_info_updated = pyqtSignal(int, int)    # (total, labeled)

    # ------------------------------------------------------------------
    # Signals: navigation
    # ------------------------------------------------------------------
    navigate_to_image = pyqtSignal(str)              # path to load
    navigate_reset = pyqtSignal(str)                 # next path after state reset (empty → no next)

    # ------------------------------------------------------------------
    # Signals: label state
    # ------------------------------------------------------------------
    label_saved = pyqtSignal()                       # view calls _set_clean()
    label_files_deleted = pyqtSignal(object, str)    # (set[deleted_paths], reload_path_or_empty)

    # ------------------------------------------------------------------
    # Signals: image reader
    # ------------------------------------------------------------------
    image_reader_clear_requested = pyqtSignal()

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    BATCH_SIZE = 250

    def __init__(
        self,
        image_index_thread_cls: type,
        project_path: str,
        session=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_index_thread_cls = image_index_thread_cls
        self._project_path = project_path
        self._session = session

        # ---- State owned by ViewModel (single source of truth) --------
        self._image_paths: list[str] = []
        self._labeled_info_dict: dict[str, bool] = {}
        self._file_list_total_count: int = 0
        self._file_list_labeled_count: int = 0
        self._pending_image_states: list = []
        self._pending_image_state_index: int = 0
        self._image_index_thread = None

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    def is_there_trained_checkpoint(self) -> bool:
        return self._session is not None and self._session.is_there_trained_checkpoint()

    # ------------------------------------------------------------------
    # Persistent image list (disk I/O)
    # ------------------------------------------------------------------

    def save_labeler_image_list(self, image_paths: list[str]) -> None:
        try:
            save_labeler_image_list(self._project_path, image_paths)
        except Exception:
            pass

    def get_labeler_image_list(self) -> list[str]:
        try:
            return load_labeler_image_list(self._project_path)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # State queries (read-only)
    # ------------------------------------------------------------------

    def get_image_paths(self) -> list[str]:
        return list(self._image_paths)

    def get_labeled_image_paths(self) -> list[str]:
        return [p for p, labeled in self._labeled_info_dict.items() if labeled]

    def get_unlabeled_image_paths(self) -> list[str]:
        return [p for p, labeled in self._labeled_info_dict.items() if not labeled]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop any running index thread; called from View.closeEvent."""
        if self._image_index_thread is not None and self._image_index_thread.isRunning():
            self._image_index_thread.request_cancel()
            self._image_index_thread.wait(3000)
        self._image_index_thread = None

    # ------------------------------------------------------------------
    # Image indexing commands
    # ------------------------------------------------------------------

    def load_images(self, img_paths: list[str]) -> None:
        """Command: start scanning and indexing a list of image paths."""
        self.image_reader_clear_requested.emit()

        if self._image_index_thread is not None and self._image_index_thread.isRunning():
            self._image_index_thread.request_cancel()
            self._image_index_thread.wait(3000)
        self._image_index_thread = None

        if not img_paths:
            self._apply_image_states([])
            return

        self.file_list_enabled_changed.emit(False)
        self.emit_status("Scanning image labels...")
        self.image_index_started.emit(len(img_paths))

        self._image_index_thread = self._image_index_thread_cls(img_paths)
        self._image_index_thread.sig_progress.connect(self._on_index_progress)
        self._image_index_thread.sig_completed.connect(self._on_index_completed)
        self._image_index_thread.sig_failed.connect(self._on_index_failed)
        self._image_index_thread.start()

    def cancel_image_index(self) -> None:
        """Command: cancel the running indexing thread."""
        if self._image_index_thread is not None:
            self._image_index_thread.request_cancel()

    # ------------------------------------------------------------------
    # Image indexing — private thread slots
    # ------------------------------------------------------------------

    def _on_index_progress(self, cur: int, total: int, path: str) -> None:
        self.image_index_progress.emit(cur, total, path)
        self.emit_status(f"Scanning image labels... ({cur}/{total})", 0)

    def _on_index_completed(self, states, was_cancelled: bool) -> None:
        self._image_index_thread = None
        self.image_index_completed.emit(was_cancelled)
        if was_cancelled:
            self.file_list_enabled_changed.emit(True)
            self.emit_status("Image indexing canceled.", 3000)
            return
        self._apply_image_states(states)

    def _on_index_failed(self, message: str) -> None:
        self._image_index_thread = None
        self.file_list_enabled_changed.emit(True)
        logger.error("Image indexing error: %s", message)
        self.image_index_failed.emit(message)
        self.emit_status("Image indexing failed.", 3000)

    # ------------------------------------------------------------------
    # Image state application (batched to keep UI responsive)
    # ------------------------------------------------------------------

    def _apply_image_states(self, states) -> None:
        from aligner_gui.labeler.libs.label_manager import LabelManager

        LabelManager.init()
        self.file_list_cleared.emit()

        for state in states:
            for label_name in state.labels:
                LabelManager.update_label_names_with_idx(label_name)

        self._image_paths = [state.path for state in states]
        self._file_list_total_count = len(self._image_paths)
        self._file_list_labeled_count = 0
        self._labeled_info_dict.clear()
        self._pending_image_states = list(states)
        self._pending_image_state_index = 0

        if not self._pending_image_states:
            self._finish_apply_image_states()
            return

        self.emit_status("Preparing image list...", 0)
        QTimer.singleShot(0, self._append_next_batch)

    def _append_next_batch(self) -> None:
        total = len(self._pending_image_states)
        if total == 0:
            self._finish_apply_image_states()
            return

        start = self._pending_image_state_index
        end = min(start + self.BATCH_SIZE, total)

        batch = []
        for state in self._pending_image_states[start:end]:
            self._labeled_info_dict[state.path] = state.has_label
            if state.has_label:
                self._file_list_labeled_count += 1
            batch.append((state.path, state.has_label, state.is_empty, state.needs_confirm))

        self.file_list_batch_ready.emit(batch)
        self._pending_image_state_index = end
        self.emit_status(f"Preparing image list... ({end}/{total})", 0)

        if end < total:
            QTimer.singleShot(0, self._append_next_batch)
        else:
            self._finish_apply_image_states()

    def _finish_apply_image_states(self) -> None:
        self._pending_image_states = []
        self._pending_image_state_index = 0
        self.file_list_enabled_changed.emit(True)
        self._emit_file_list_info()
        self.emit_status("Image indexing finished.", 3000)
        if self._image_paths:
            self.navigate_to_image.emit(self._image_paths[0])

    # ------------------------------------------------------------------
    # Navigation commands
    # ------------------------------------------------------------------

    def navigate_to_previous(self, current_path: str | None) -> None:
        """Command: navigate to the image before current_path."""
        if not self._image_paths or current_path is None:
            return
        try:
            idx = self._image_paths.index(current_path)
        except ValueError:
            return
        if idx - 1 >= 0:
            self.navigate_to_image.emit(self._image_paths[idx - 1])

    def navigate_to_next(self, current_path: str | None) -> None:
        """Command: navigate to the image after current_path."""
        if not self._image_paths:
            return
        if current_path is None:
            self.navigate_to_image.emit(self._image_paths[0])
            return
        try:
            idx = self._image_paths.index(current_path)
        except ValueError:
            return
        if idx + 1 < len(self._image_paths):
            self.navigate_to_image.emit(self._image_paths[idx + 1])

    # ------------------------------------------------------------------
    # Label save commands
    # ------------------------------------------------------------------

    def save_label(
        self,
        image_path: str,
        shapes,
        width: int,
        height: int,
        is_grayscale: bool,
    ) -> None:
        """Command: save label data for the current image."""
        if image_path is None:
            return
        label_file_class = self._get_label_file_class()
        label_file = label_file_class(image_path)
        label_file.set_shapes(shapes)
        label_file.save_label(image_info={
            "height": height,
            "width": width,
            "depth": 1 if is_grayscale else 3,
            "isNeedConfirm": False,
        })
        self.label_saved.emit()
        current_shapes = label_file.get_shapes()
        self.mark_path_as_saved(
            image_path,
            has_label=True,
            is_empty=(len(current_shapes) == 0),
            needs_confirm=False,
        )

    def mark_path_as_saved(
        self,
        image_path: str,
        has_label: bool,
        is_empty: bool = False,
        needs_confirm: bool = False,
    ) -> None:
        """Update in-memory state and emit signals after a label save."""
        was_labeled = self._labeled_info_dict.get(image_path, False)
        if has_label and not was_labeled:
            self._file_list_labeled_count += 1
        elif not has_label and was_labeled:
            self._file_list_labeled_count = max(0, self._file_list_labeled_count - 1)
        self._labeled_info_dict[image_path] = has_label
        self.file_list_item_state_changed.emit(image_path, has_label, is_empty, needs_confirm)
        self._emit_file_list_info()

    # ------------------------------------------------------------------
    # Label delete command
    # ------------------------------------------------------------------

    def delete_label_files(
        self,
        target_paths: list[str],
        current_image_path: str | None,
    ) -> None:
        """Command: delete label files for the given paths."""
        label_file_class = self._get_label_file_class()
        deleted_paths: set[str] = set()
        for target_path in target_paths:
            label_file = label_file_class(target_path)
            label_file.remove_label_file()
            if self._labeled_info_dict.get(target_path, False):
                self._labeled_info_dict[target_path] = False
                self._file_list_labeled_count -= 1
            self.file_list_item_state_changed.emit(target_path, False, False, False)
            deleted_paths.add(target_path)
        self._emit_file_list_info()
        reload_path = current_image_path if current_image_path in deleted_paths else ""
        self.label_files_deleted.emit(deleted_paths, reload_path)

    # ------------------------------------------------------------------
    # Remove images from list command
    # ------------------------------------------------------------------

    def remove_images_from_list(
        self,
        target_paths: list[str],
        current_image_path: str | None,
    ) -> None:
        """Command: remove images from the project list without deleting files."""
        removed_paths = set(target_paths)
        removal_result = remove_paths_from_file_list(
            self._image_paths, self._labeled_info_dict, removed_paths, current_image_path
        )
        self._image_paths = removal_result.image_paths
        self._labeled_info_dict = removal_result.labeled_info
        self._file_list_labeled_count = removal_result.labeled_count
        self._file_list_total_count = len(removal_result.image_paths)
        self.save_labeler_image_list(self._image_paths)
        self.file_list_items_removed.emit(removed_paths)
        self._emit_file_list_info()
        if removal_result.removed_current:
            self.navigate_reset.emit(removal_result.next_image_path or "")
        self.emit_status(
            f"Removed {removal_result.removed_count} image(s) from the project list.", 3000
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_file_list_info(self) -> None:
        self.file_list_info_updated.emit(self._file_list_total_count, self._file_list_labeled_count)

    @staticmethod
    def _get_label_file_class():
        from aligner_gui.labeler.libs.labelFile import LabelFile
        return LabelFile
