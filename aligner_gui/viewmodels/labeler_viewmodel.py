from __future__ import annotations

import logging
import os
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox, QProgressDialog

from aligner_gui import __appname__
from aligner_gui.labeler.file_list_service import remove_paths_from_file_list
from aligner_gui.project.project_dataset_service import load_labeler_image_list, save_labeler_image_list
from aligner_gui.shared import gui_util
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase

if TYPE_CHECKING:
    from aligner_gui.labeler.labeler_view import LabelerView, ImageIndexThread
    from aligner_gui.labeler.libs.shape import Shape


class LabelerViewModel(ViewModelBase):
    def __init__(self, view: 'LabelerView', image_index_thread_cls: type['ImageIndexThread'], session=None):
        super().__init__(view)
        self.view = view
        self._image_index_thread_cls = image_index_thread_cls
        self._session = session

    def is_there_trained_checkpoint(self) -> bool:
        return self._session is not None and self._session.is_there_trained_checkpoint()

    def set_labeler_image_list_to_file(self, image_paths):
        try:
            save_labeler_image_list(self.view._project_path, image_paths)
        except Exception:
            pass

    def get_labeler_image_list_from_file(self):
        try:
            return load_labeler_image_list(self.view._project_path)
        except Exception:
            return []

    def load_images(self, img_paths):
        self.view._image_reader.clear()
        if self.view._image_index_thread is not None and self.view._image_index_thread.isRunning():
            self.view._image_index_thread.request_cancel()
            self.view._image_index_thread.wait(3000)
        if self.view._image_index_progress is not None:
            self.view._image_index_progress.close()
            self.view._image_index_progress.deleteLater()
            self.view._image_index_progress = None

        if len(img_paths) == 0:
            self.apply_image_states([])
            return

        self.view._file_list_widget.setEnabled(False)
        self.emit_status("Scanning image labels...")

        self.view._image_index_thread = self._image_index_thread_cls(img_paths)
        self.view._image_index_thread.sig_progress.connect(self.progress_image_index)
        self.view._image_index_thread.sig_completed.connect(self.completed_image_index)
        self.view._image_index_thread.sig_failed.connect(self.failed_image_index)

        if self.view.isVisible():
            self.view._image_index_progress = QProgressDialog("Scanning image labels...", "Cancel", 0, len(img_paths), self.view)
            self.view._image_index_progress.setWindowTitle("Indexing Images")
            self.view._image_index_progress.setWindowModality(Qt.WindowModal)
            self.view._image_index_progress.setMinimumDuration(0)
            self.view._image_index_progress.setValue(0)
            self.view._image_index_progress.canceled.connect(self.view._image_index_thread.request_cancel)
            self.view._image_index_progress.show()
        else:
            self.view._image_index_progress = None

        self.view._image_index_thread.start()

    def progress_image_index(self, cur_idx: int, total: int, path: str):
        progress = self.view._image_index_progress
        if progress is not None:
            try:
                progress.setMaximum(max(total, 1))
                progress.setLabelText(f"Scanning image labels...\n{os.path.basename(path)}")
                progress.setValue(cur_idx)
            except RuntimeError:
                self.view._image_index_progress = None
        self.emit_status(f"Scanning image labels... ({cur_idx}/{total})", 0)

    def completed_image_index(self, states, was_cancelled: bool):
        if self.view._image_index_progress is not None:
            self.view._image_index_progress.close()
            self.view._image_index_progress.deleteLater()
            self.view._image_index_progress = None

        self.view._image_index_thread = None
        if was_cancelled:
            self.view._file_list_widget.setEnabled(True)
            self.emit_status("Image indexing canceled.", 3000)
            return

        self.apply_image_states(states)

    def failed_image_index(self, message: str):
        if self.view._image_index_progress is not None:
            self.view._image_index_progress.close()
            self.view._image_index_progress.deleteLater()
            self.view._image_index_progress = None

        self.view._image_index_thread = None
        self.view._file_list_widget.setEnabled(True)
        logging.error("ERROR - %s", message)
        gui_util.get_message_box(self.view, "Indexing Failed", "Failed to scan image labels.")
        self.emit_status("Image indexing failed.", 3000)

    def apply_image_states(self, states):
        from aligner_gui.labeler.libs.label_manager import LabelManager

        LabelManager.init()
        self.view._file_list_widget.clear()

        for state in states:
            for label_name in state.labels:
                LabelManager.update_label_names_with_idx(label_name)

        self.view._image_paths = [state.path for state in states]
        self.view._file_list_total_count = len(self.view._image_paths)
        self.view._file_list_labeled_count = 0
        self.view._labeled_info_dict.clear()
        self.view._pending_image_states = list(states)
        self.view._pending_image_state_index = 0
        if len(self.view._pending_image_states) == 0:
            self.finish_apply_image_states()
            return

        self.emit_status("Preparing image list...", 0)
        QTimer.singleShot(0, self.append_next_image_state_batch)

    def append_next_image_state_batch(self):
        total = len(self.view._pending_image_states)
        if total == 0:
            self.finish_apply_image_states()
            return
        start = self.view._pending_image_state_index
        end = min(start + self.view.IMAGE_LIST_BATCH_SIZE, total)
        self.view._file_list_widget.setUpdatesEnabled(False)
        for state in self.view._pending_image_states[start:end]:
            item = QListWidgetItem(state.path)
            self.set_file_item_state(item, state.path, state.has_label, state.is_empty, state.needs_confirm)
            if state.has_label:
                self.view._file_list_labeled_count += 1
            self.view._file_list_widget.addItem(item)
        self.view._file_list_widget.setUpdatesEnabled(True)
        self.view._pending_image_state_index = end
        self.emit_status(f"Preparing image list... ({end}/{total})", 0)
        if end < total:
            QTimer.singleShot(0, self.append_next_image_state_batch)
            return
        self.finish_apply_image_states()

    def finish_apply_image_states(self):
        self.view._pending_image_states = []
        self.view._pending_image_state_index = 0
        self.view._file_list_widget.setEnabled(True)
        self.refresh_file_list_info()
        self.emit_status("Image indexing finished.", 3000)
        self.view.queueEvent(self.open_next_image)

    def open_previous_image(self, _value=False):
        if not self.view.mayContinue() or len(self.view._image_paths) <= 0 or self.view._current_image_path is None:
            return
        curr_index = self.view._image_paths.index(self.view._current_image_path)
        if curr_index - 1 >= 0:
            filename = self.view._image_paths[curr_index - 1]
            if filename:
                self.view._load_file(filename)

    def open_next_image(self, _value=False):
        if not self.view.mayContinue() or len(self.view._image_paths) <= 0:
            return
        filename = None
        if self.view._current_image_path is None:
            filename = self.view._image_paths[0]
        else:
            curr_index = self.view._image_paths.index(self.view._current_image_path)
            if curr_index + 1 < len(self.view._image_paths):
                filename = self.view._image_paths[curr_index + 1]
        if filename:
            self.view._load_file(filename)

    def open_file(self, _value=False):
        if not self.view.mayContinue():
            return
        path = os.path.dirname(self.view._current_image_path) if self.view._current_image_path else '.'
        formats = gui_util.SUPPORTED_IMAGE_FORMATS
        label_file_class = self.view._get_label_file_class()
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % label_file_class.SUFFIX])
        filename = QFileDialog.getOpenFileName(self.view, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.view._load_file(filename)

    def open_dir(self, _value=False):
        if not self.view.mayContinue():
            return
        path = os.path.dirname(self.view._current_image_path) if self.view._current_image_path else '.'
        if self.view.lastOpenDir is not None and len(self.view.lastOpenDir) > 1:
            path = self.view.lastOpenDir
        dirpath = self.view.tr(str(QFileDialog.getExistingDirectory(
            self.view,
            '%s - Open Directory' % __appname__,
            path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )))
        if dirpath == "":
            return
        self.view.lastOpenDir = dirpath
        self.view._dir_name = dirpath
        self.view._current_image_path = None
        image_paths = self.view.scanAllImages(dirpath)
        self.set_labeler_image_list_to_file(image_paths)
        self.load_images(image_paths)

    def save_label(self, _value=False):
        if self.view._current_image_path is None or self.view._current_image.isNull():
            return
        label_file_class = self.view._get_label_file_class()
        label_file = label_file_class(self.view._current_image_path)
        label_file.set_shapes(self.view.canvas.get_shape())
        label_file.save_label(image_info={
            "height": self.view._current_image.height(),
            "width": self.view._current_image.width(),
            "depth": 1 if self.view._current_image.isGrayscale() else 3,
            "isNeedConfirm": False,
        })
        self.view._set_clean()
        current_shapes = label_file.get_shapes()
        self.mark_path_as_saved(self.view._current_image_path, has_label=True, is_empty=(len(current_shapes) == 0), needs_confirm=False)

    def save_selected_labels(self, _value=False):
        target_paths = self.get_selected_image_paths()
        if not target_paths or self.view._current_image_path is None:
            return
        selected_shapes = deepcopy(self.view.canvas.get_shape())
        saved_paths = []
        failed_paths = []
        if len(target_paths) > 1:
            msg = self.view.tr(
                "Save current labels to {} selected images?\r\n"
                "Existing label files of those images will be overwritten."
            ).format(len(target_paths))
            if QMessageBox.warning(self.view, self.view.tr("Attention"), msg, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        def work(idx: int):
            target_path = target_paths[idx]
            try:
                self.save_shapes_to_image_path(target_path, selected_shapes)
                saved_paths.append(target_path)
            except Exception:
                logging.exception("Failed to save labels to %s", target_path)
                failed_paths.append(target_path)

        progress_dialog_class = self.view._get_progress_list_dialog_class()
        dlg = progress_dialog_class(work, target_paths)
        dlg.exec_()
        for target_path in saved_paths:
            self.mark_path_as_saved(target_path, has_label=True, is_empty=(len(selected_shapes) == 0), needs_confirm=False)
        if self.view._current_image_path in saved_paths:
            self.view._set_clean()
        if failed_paths:
            gui_util.get_message_box(
                self.view,
                "Batch Save",
                f"Saved {len(saved_paths)} image(s).\nFailed: {len(failed_paths)} image(s). Check the log for details.",
            )
        self.emit_status(f"Saved labels to {len(saved_paths)} image(s).", 3000)

    def delete_label_file(self):
        selected_items = self.view._file_list_widget.selectedItems()
        target_paths = [self.view.tr(item.text()) for item in selected_items]
        if not target_paths and self.view._current_image_path is not None:
            target_paths = [self.view._current_image_path]
        if not target_paths:
            return
        mb = QMessageBox
        if len(target_paths) == 1:
            msg = self.view.tr("You are about to permanently delete label file of {}, \r\nproceed anyway?").format(target_paths[0])
        else:
            msg = self.view.tr("You are about to permanently delete {} label files, \r\nproceed anyway?").format(len(target_paths))
        answer = mb.warning(self.view, self.view.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return
        label_file_class = self.view._get_label_file_class()
        current_image_path = self.view._current_image_path
        current_deleted = current_image_path in target_paths
        for target_path in target_paths:
            label_file = label_file_class(target_path)
            label_file.remove_label_file()
            if self.view._labeled_info_dict.get(target_path, False):
                self.view._labeled_info_dict[target_path] = False
                self.view._file_list_labeled_count -= 1
            self.set_file_item_state_by_path(target_path, has_label=False)
        self.refresh_file_list_info()
        if current_deleted and current_image_path is not None:
            self.view.resetState()
            self.view._load_file(current_image_path, is_load_after_delete=True)

    def remove_selected_images_from_list(self):
        selected_items = self.view._file_list_widget.selectedItems()
        target_paths = [self.view.tr(item.text()) for item in selected_items]
        if not target_paths:
            return
        mb = QMessageBox
        if len(target_paths) == 1:
            msg = self.view.tr("You are about to remove {} from the current project list.\r\nThe image file itself will not be deleted. Proceed anyway?").format(target_paths[0])
        else:
            msg = self.view.tr("You are about to remove {} images from the current project list.\r\nThe image files themselves will not be deleted. Proceed anyway?").format(len(target_paths))
        answer = mb.warning(self.view, self.view.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return
        removed_paths = set(target_paths)
        current_path = self.view._current_image_path
        removal_result = remove_paths_from_file_list(self.view._image_paths, self.view._labeled_info_dict, removed_paths, current_path)
        for row in range(self.view._file_list_widget.count() - 1, -1, -1):
            item = self.view._file_list_widget.item(row)
            item_path = self.view.tr(item.text())
            if item_path not in removed_paths:
                continue
            self.view._file_list_widget.takeItem(row)
        self.view._image_paths = removal_result.image_paths
        self.view._labeled_info_dict = removal_result.labeled_info
        self.view._file_list_labeled_count = removal_result.labeled_count
        self.view._file_list_total_count = len(removal_result.image_paths)
        self.set_labeler_image_list_to_file(self.view._image_paths)
        self.refresh_file_list_info()
        if removal_result.removed_current:
            self.view.resetState()
            self.view._set_clean()
            self.view.canvas.setEnabled(False)
            self.view._current_image_path = None
            self.view._current_image = QImage()
            if removal_result.next_image_path is not None:
                self.view._load_file(removal_result.next_image_path)
            else:
                self.view._main_window.setWindowTitle(__appname__)
        self.emit_status(f"Removed {removal_result.removed_count} image(s) from the project list.", 3000)

    def refresh_file_list_info(self):
        file_list_info = 'Total: {0}, Labeled: {1}'.format(self.view._file_list_total_count, self.view._file_list_labeled_count)
        self.view._lbl_file_list_info.setText(file_list_info)

    def get_selected_image_paths(self):
        selected_items = self.view._file_list_widget.selectedItems()
        target_paths = [self.view.tr(item.text()) for item in selected_items]
        if not target_paths and self.view._current_image_path is not None:
            target_paths = [self.view._current_image_path]
        return list(dict.fromkeys(target_paths))

    def select_all_files_in_list(self):
        if self.view._file_list_widget.count() == 0:
            return
        self.view._file_list_widget.setFocus(Qt.ShortcutFocusReason)
        self.view._file_list_widget.selectAll()
        self.emit_status(f"Selected {self.view._file_list_widget.count()} images.", 2000)

    def set_file_item_state(self, item: QListWidgetItem, image_path: str, has_label: bool, is_empty: bool = False, needs_confirm: bool = False):
        self.view._labeled_info_dict[image_path] = has_label
        if not has_label:
            foreground = QColor(235, 92, 92)
            background = QColor(62, 24, 28)
        elif is_empty:
            foreground = QColor(160, 166, 176)
            background = QColor(38, 42, 48)
        elif needs_confirm:
            foreground = QColor(255, 176, 245)
            background = QColor(54, 34, 60)
        else:
            foreground = QColor(232, 236, 241)
            background = QColor(30, 34, 38)
        item.setForeground(foreground)
        item.setBackground(background)

    def set_file_item_state_by_path(self, image_path: str, has_label: bool, is_empty: bool = False, needs_confirm: bool = False):
        for item in self.view._file_list_widget.findItems(image_path, Qt.MatchExactly):
            self.set_file_item_state(item, image_path, has_label, is_empty, needs_confirm)

    def mark_path_as_saved(self, image_path: str, has_label: bool, is_empty: bool = False, needs_confirm: bool = False):
        was_labeled = self.view._labeled_info_dict.get(image_path, False)
        if has_label and not was_labeled:
            self.view._file_list_labeled_count += 1
        elif not has_label and was_labeled:
            self.view._file_list_labeled_count = max(0, self.view._file_list_labeled_count - 1)
        self.set_file_item_state_by_path(image_path, has_label, is_empty, needs_confirm)
        self.refresh_file_list_info()

    def save_shapes_to_image_path(self, image_path: str, shapes: list['Shape']):
        label_file_class = self.view._get_label_file_class()
        label_file = label_file_class(image_path)
        label_file.set_shapes(deepcopy(shapes))
        if image_path == self.view._current_image_path and not self.view._current_image.isNull():
            current_image = self.view._current_image
        else:
            current_image = self.view.get_qimage_from_mat(self.view._read_image_mat(image_path))
        if current_image.isNull():
            raise RuntimeError(f"Failed to read image: {image_path}")
        label_file.save_label(image_info={
            "height": current_image.height(),
            "width": current_image.width(),
            "depth": 1 if current_image.isGrayscale() else 3,
            "isNeedConfirm": False,
        })

