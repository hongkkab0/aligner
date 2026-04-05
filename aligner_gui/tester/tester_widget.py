from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import List

from PyQt5.QtCore import Qt, QPoint, pyqtSlot
from PyQt5.QtGui import QColor, QMovie, QPen, QCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QHeaderView,
    QMenu,
    QSignalBlocker,
    QSignalMapper,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QTableWidgetItem,
    QWidget,
)
from PyQt5 import QtGui
import logging
import numpy as np

from aligner_gui.ui.tester_widget import Ui_tester_widget
from aligner_gui.tester.preview_renderer import TesterPreviewRenderer
from aligner_gui.utils import const, gui_util
from aligner_gui.utils import io_util
from aligner_gui.widgets.image_panel import ImagePanel
from aligner_gui.viewmodels.tester_viewmodel import TesterViewModel

TEST_LOGGER = logging.getLogger("aligner.tester")


class TesterWidget(QWidget, Ui_tester_widget):
    """View layer for the Tester tab.

    All business logic lives in :class:`~aligner_gui.viewmodels.tester_viewmodel.TesterViewModel`.
    This class is responsible only for:

    * Building and wiring up the Qt UI elements.
    * Connecting ViewModel signals to local UI-update slots.
    * Delegating user actions to the ViewModel via command methods.
    * Showing file/folder dialogs and context menus.
    """

    COLOR_INFERENCE = """QProgressBar::chunk { background: yellow; }"""

    def __init__(self, session, is_new: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self._vm = TesterViewModel(session, parent=self)

        # -- Icons
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\plus(white).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_add.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\folder-10(white).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_add_folder.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\garbage-2(white).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_delete.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\075-reload.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_reset.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\083-share.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_export_csv.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\save(white).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_save_image.setIcon(icon)

        # -- Button connections (View → ViewModel commands)
        self.btn_add.clicked.connect(self._clicked_btn_add)
        self.btn_add_folder.clicked.connect(self._clicked_btn_add_folder)
        self.btn_delete.clicked.connect(self._clicked_btn_delete)
        self.btn_reset.clicked.connect(self._clicked_btn_reset)
        self.btn_export_csv.clicked.connect(self._clicked_btn_export_csv)
        self.btn_save_image.clicked.connect(self._clicked_btn_save_image)
        self.btn_test.clicked.connect(self._clicked_btn_test)

        # -- Connect ViewModel signals → View slots
        self._vm.testing_started.connect(self._on_testing_started)
        self._vm.testing_stopped.connect(self._on_testing_stopped)
        self._vm.iter_updated.connect(self._on_iter_updated)
        self._vm.results_updated.connect(self._on_results_updated)
        self._vm.file_list_changed.connect(self._on_file_list_changed)

        # -- Busy indicator
        self._busy_indicator = QMovie("aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif")
        self.lbl_test_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_test_indicator.hide()

        # -- Image panel
        self.image_panel = ImagePanel(self)
        self.image_panel.setROIMode(False)
        self.image_panel.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.layout_test_image_viewer.addWidget(self.image_panel)
        self.splitter_result.setStretchFactor(1, 1)

        self.check_show_gt = QCheckBox("GT")
        self.check_show_gt.setChecked(True)
        self.check_show_gt.stateChanged.connect(self._refresh_current_preview)
        self.horizontalLayout_2.insertWidget(1, self.check_show_gt)

        self.check_show_prediction = QCheckBox("Pred")
        self.check_show_prediction.setChecked(True)
        self.check_show_prediction.stateChanged.connect(self._refresh_current_preview)
        self.horizontalLayout_2.insertWidget(2, self.check_show_prediction)

        # -- Detail table
        self._delegate_table_test_detail = ElideLeftDelegate()
        self.table_test_detail.setItemDelegate(self._delegate_table_test_detail)
        self.table_test_detail.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_test_detail.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_test_detail.itemSelectionChanged.connect(self._item_selected_changed_table_detail)
        self.table_test_detail.cellPressed.connect(self._cell_pressed_table_detail)
        self.table_test_detail.horizontalHeader().sectionClicked.connect(self._section_clicked_table_detail)
        self._src_path_clicked_of_table_detail: str = ""
        self.table_test_detail_sort_col: int = -1

        # -- File list table
        self._delegate_table_file_list = ElideLeftDelegate()
        self.table_file_list.setItemDelegate(self._delegate_table_file_list)
        self.table_file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_file_list.itemSelectionChanged.connect(self._item_selected_changed_table_file_list)

        # -- Preview renderer (View concern: knows about cv2 and rendering)
        self._preview_renderer = TesterPreviewRenderer(self._get_cv2, self._vm.get_project_settings)
        self._last_preview_key = None

        if not is_new:
            self._vm.reload_file_list()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_torch(self):
        import torch
        return torch

    def _get_cv2(self):
        import cv2
        return cv2

    def _clear_preview_cache(self) -> None:
        self._preview_renderer.clear()
        self._last_preview_key = None

    # ------------------------------------------------------------------
    # Button handlers (View → ViewModel)
    # ------------------------------------------------------------------

    def _clicked_btn_test(self) -> None:
        if self.btn_test.isChecked():
            self._vm.start_testing()
        else:
            self._vm.stop_testing("manual stop")

    def _clicked_btn_add(self) -> None:
        extensions = gui_util.SUPPORTED_IMAGE_FORMATS_WITHOUT_DOT
        file_paths = gui_util.get_files_from_dialog(self, is_open=True, ext_list=extensions)
        if not file_paths:
            return
        self._clear_preview_cache()
        self._vm.add_files(file_paths)

    def _clicked_btn_add_folder(self) -> None:
        dir_path = gui_util.get_open_dir_from_dialog(self, "Choose a directory to test images")
        if not dir_path:
            return
        image_paths = self._scan_all_images(dir_path)
        self._clear_preview_cache()
        self._vm.add_files(image_paths)

    def _clicked_btn_delete(self) -> None:
        selected_indexes = self.table_file_list.selectionModel().selectedIndexes()
        rows = sorted(
            {idx.row() for idx in selected_indexes if idx.column() == 0},
            reverse=True,
        )
        if not rows:
            return
        self._clear_preview_cache()
        self._vm.remove_files_at_rows(rows)

    def _clicked_btn_reset(self) -> None:
        self._clear_preview_cache()
        self._vm.reload_file_list()

    def _clicked_btn_save_image(self) -> None:
        extensions = ["bmp", "png", "jpg"]
        cur_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_file_name = "Image__%s.jpg" % cur_time_str
        file_path = gui_util.get_file_from_dialog(self, is_open=False, ext_list=extensions, save_file_name=save_file_name)
        if not file_path:
            return
        self.image_panel.saveImage(file_path)

    def _clicked_btn_export_csv(self) -> None:
        if self.table_test_detail.rowCount() <= 0:
            return
        save_path = gui_util.get_path_from_dialog(self, "Export test result")
        cur_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "[dice_aligner_test_result] %s.csv" % cur_time_str
        file_name = io_util.join_path(save_path, file_name)
        gui_util.table2csv(self.table_test_detail, file_name)
        gui_util.get_message_box(self, "Export test result", "Test result is saved to %s." % file_name)

    def _scan_all_images(self, folder_path: str) -> List[str]:
        extensions = gui_util.SUPPORTED_IMAGE_FORMATS
        images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    path = os.path.abspath(os.path.join(root, file))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    # ------------------------------------------------------------------
    # ViewModel signal slots
    # ------------------------------------------------------------------

    def _on_testing_started(self) -> None:
        self.btn_test.setChecked(True)
        self.btn_test.setEnabled(False)
        self.progress_iter.setMaximum(1)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.lbl_test_indicator.show()

    def _on_testing_stopped(self, reason: str) -> None:
        torch = self._get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reason == const.SUCCESS:
            TEST_LOGGER.info("test finished successfully")
        elif reason == "no_checkpoint":
            gui_util.get_message_box(self, "Invalid Test", "There is no trained model.")
        elif reason == const.ERROR:
            TEST_LOGGER.error("test failed")
        else:
            TEST_LOGGER.info(reason)

        self.btn_test.setChecked(False)
        self.btn_test.setEnabled(True)
        self.progress_iter.setMaximum(1)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.lbl_test_indicator.hide()

    def _on_iter_updated(self, iter_idx: int, iter_len: int) -> None:
        self.progress_iter.setStyleSheet(self.COLOR_INFERENCE)
        self.progress_iter.setMaximum(iter_len)
        self.progress_iter.setValue(iter_idx + 1)
        self.btn_test.setEnabled(True)

    def _on_results_updated(self) -> None:
        self._clear_preview_cache()
        self._refresh_test_detail_table()
        self._refresh_test_time()

    def _on_file_list_changed(self, file_list: List[str]) -> None:
        self._refresh_table_file_list(file_list)

    # ------------------------------------------------------------------
    # Preview rendering (View-only)
    # ------------------------------------------------------------------

    def _refresh_current_preview(self) -> None:
        self._last_preview_key = None
        indexes = self.table_test_detail.selectedIndexes()
        if indexes:
            idx = indexes[0].row()
            img_path = self.table_test_detail.item(idx, 0).text()
            self._change_selected_image_table_detail(img_path)
            return

        indexes = self.table_file_list.selectedIndexes()
        if indexes:
            idx = indexes[0].row()
            img_path = self.table_file_list.item(idx, 0).text()
            self._change_selected_image_table_file_list(img_path)

    def _change_selected_image_table_file_list(self, img_path: str) -> None:
        preview_key = ("file", img_path)
        if self._last_preview_key == preview_key:
            return
        qpixmap = self._preview_renderer.render_file_preview(img_path)
        if qpixmap is None:
            return
        self.image_panel.setImage(qpixmap)
        self._last_preview_key = preview_key

    def _change_selected_image_table_detail(self, img_path: str) -> None:
        preview_key = (
            "detail", img_path,
            self.check_show_gt.isChecked(),
            self.check_show_prediction.isChecked(),
        )
        if self._last_preview_key == preview_key:
            return
        summary = self._vm.test_result_summary
        if summary is None:
            return
        qpixmap = self._preview_renderer.render_detail_preview(
            img_path,
            summary,
            self._vm.class_index,
            self.check_show_gt.isChecked(),
            self.check_show_prediction.isChecked(),
        )
        if qpixmap is None:
            return
        self.image_panel.setImage(qpixmap)
        self._last_preview_key = preview_key

    # ------------------------------------------------------------------
    # Table updates
    # ------------------------------------------------------------------

    def _refresh_table_file_list(self, file_list: List[str]) -> None:
        selected_paths = {
            self.table_file_list.item(index.row(), 0).text()
            for index in self.table_file_list.selectedIndexes()
            if index.column() == 0 and self.table_file_list.item(index.row(), 0) is not None
        }
        self._last_preview_key = None
        table_blocker = QSignalBlocker(self.table_file_list)
        self.table_file_list.setUpdatesEnabled(False)
        self.table_file_list.clear()
        column_names = ["Path", "Label"]
        self.table_file_list.setColumnCount(len(column_names))
        self.table_file_list.setHorizontalHeaderLabels(column_names)
        self.table_file_list.setColumnWidth(0, 150)
        self.table_file_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_file_list.setRowCount(len(file_list))
        for idx, img_path in enumerate(file_list):
            self.table_file_list.setItem(idx, 0, QTableWidgetItem(img_path))
            label_exists = os.path.exists(os.path.splitext(img_path)[0] + ".json")
            self.table_file_list.setItem(idx, 1, QTableWidgetItem("O" if label_exists else "X"))
            if img_path in selected_paths:
                self.table_file_list.selectRow(idx)
        self.table_file_list.setUpdatesEnabled(True)
        del table_blocker

    def _refresh_test_detail_table(self) -> None:
        summary = self._vm.test_result_summary
        if summary is None:
            return

        selected_img_path = None
        selected_indexes = self.table_test_detail.selectedIndexes()
        if selected_indexes:
            item = self.table_test_detail.item(selected_indexes[0].row(), 0)
            if item is not None:
                selected_img_path = item.text()

        table_blocker = QSignalBlocker(self.table_test_detail)
        self.table_test_detail.setUpdatesEnabled(False)
        self.table_test_detail.clear()
        self.table_test_detail.setSortingEnabled(False)

        data_results = summary.data_result
        max_roi = 1
        for data_result in data_results.values():
            if len(data_result) > max_roi:
                max_roi = len(data_result)

        column_names = [
            "Path", "Corner\nError", "Corner\nX", "Corner\nY",
            "Center\nError", "Center\nX", "Center\nY",
            "Longside\nError", "Shortside\nError", "# ROI",
        ] + ["R%d" % i for i in range(max_roi)]

        self.table_test_detail.setColumnCount(len(column_names))
        self.table_test_detail.setHorizontalHeaderLabels(column_names)
        self.table_test_detail.setColumnWidth(0, 100)
        for i in range(1, len(column_names)):
            self.table_test_detail.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table_test_detail.setRowCount(len(data_results))

        for idx, (img_path, data_result) in enumerate(data_results.items()):
            self.table_test_detail.setItem(idx, 0, QTableWidgetItem(img_path))

            metric_cols = [
                (1, "corner_error"), (2, "corner_dx"), (3, "corner_dy"),
                (4, "center_error"), (5, "center_dx"), (6, "center_dy"),
                (7, "longside"), (8, "shortside"),
            ]
            for col, key in metric_cols:
                diffs = [data_result[str(i)][key] for i in range(len(data_result)) if key in data_result[str(i)]]
                value = "%.2lf" % np.asarray(diffs).mean() if diffs else "-"
                self.table_test_detail.setItem(idx, col, QTableWidgetItemNum(value))

            self.table_test_detail.setItem(idx, 9, QTableWidgetItemNum(str(len(data_result))))

            for i in range(len(data_result)):
                self.table_test_detail.setItem(
                    idx, 10 + i,
                    QTableWidgetItem("%s (%.2lf)" % (data_result[str(i)]["class_name"], data_result[str(i)]["conf"])),
                )

        if selected_img_path is not None:
            for row in range(self.table_test_detail.rowCount()):
                item = self.table_test_detail.item(row, 0)
                if item is not None and item.text() == selected_img_path:
                    self.table_test_detail.selectRow(row)
                    break

        self.table_test_detail.setUpdatesEnabled(True)
        del table_blocker

    def _refresh_test_time(self) -> None:
        self.label_time.setText("%.3fsec/image" % self._vm.mean_test_time)

    # ------------------------------------------------------------------
    # Table interaction handlers
    # ------------------------------------------------------------------

    def _item_selected_changed_table_detail(self) -> None:
        indexes = self.table_test_detail.selectedIndexes()
        if indexes:
            img_path = self.table_test_detail.item(indexes[0].row(), 0).text()
            self._change_selected_image_table_detail(img_path)

    def _item_selected_changed_table_file_list(self) -> None:
        indexes = self.table_file_list.selectedIndexes()
        if indexes:
            img_path = self.table_file_list.item(indexes[0].row(), 0).text()
            self._change_selected_image_table_file_list(img_path)

    @pyqtSlot(int, int)
    def _cell_pressed_table_detail(self, idx: int, col: int) -> None:
        if self.table_test_detail.rowCount() < 1 or len(self.table_test_detail.selectedItems()) != 1:
            return
        if col == 0 and QApplication.mouseButtons() & Qt.RightButton:
            self._src_path_clicked_of_table_detail = (
                self.table_test_detail.item(idx, col).text().replace("\\", "/")
            )
            menu = QMenu(self)
            open_src_path_action = QAction("Open Source File", self)
            open_src_path_action.triggered.connect(self.on_open_src_path_triggered)
            menu.addAction(open_src_path_action)
            open_src_folder_action = QAction("Open Source Folder", self)
            open_src_folder_action.triggered.connect(self.on_open_src_folder_triggered)
            menu.addAction(open_src_folder_action)
            menu.exec_(QPoint(QCursor.pos().x() + 2, QCursor.pos().y() + 2))

    def on_open_src_path_triggered(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            os.startfile(self._src_path_clicked_of_table_detail)
        except Exception as e:
            print(e)
        QApplication.restoreOverrideCursor()

    def on_open_src_folder_triggered(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            subprocess.Popen(
                r'explorer /select,"' + os.path.realpath(self.table_test_detail.selectedItems()[0].text()) + '"'
            )
        except Exception as e:
            print(e)
        QApplication.restoreOverrideCursor()

    @pyqtSlot(int)
    def _section_clicked_table_detail(self, col: int) -> None:
        if col in range(1, 10):
            self.table_test_detail.setSortingEnabled(False)
            self.table_test_detail_sort_col = col
            menu = QMenu(self)
            sort_ascending = QAction("▲", self)
            sort_ascending.triggered.connect(self._on_sort_ascending)
            sort_descending = QAction("▼", self)
            sort_descending.triggered.connect(self._on_sort_descending)
            menu.addAction(sort_ascending)
            menu.addAction(sort_descending)
            header_pos = self.table_test_detail.mapToGlobal(self.table_test_detail.horizontalHeader().pos())
            pos_y = header_pos.y() + self.table_test_detail.horizontalHeader().height()
            pos_x = (
                header_pos.x()
                + self.table_test_detail.horizontalHeader().sectionViewportPosition(self.table_test_detail_sort_col)
            )
            menu.exec_(QPoint(pos_x, pos_y))
        else:
            self.table_test_detail.setSortingEnabled(False)

    def _on_sort_ascending(self) -> None:
        self.table_test_detail.setSortingEnabled(True)
        self.table_test_detail.sortByColumn(self.table_test_detail_sort_col, Qt.AscendingOrder)

    def _on_sort_descending(self) -> None:
        self.table_test_detail.setSortingEnabled(True)
        self.table_test_detail.sortByColumn(self.table_test_detail_sort_col, Qt.DescendingOrder)

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def reload_file_list(self) -> None:
        self._vm.reload_file_list()

    # ------------------------------------------------------------------
    # QWidget lifecycle
    # ------------------------------------------------------------------

    def close(self) -> bool:
        self._vm.close()
        self._preview_renderer.clear()
        return super().close()


# ---------------------------------------------------------------------------
# Helper delegates / item types
# ---------------------------------------------------------------------------

class QTableWidgetItemNum(QTableWidgetItem):
    def __lt__(self, other) -> bool:
        self_val = -1.0 if self.text() == "-" else float(self.text())
        other_val = -1.0 if other.text() == "-" else float(other.text())
        return self_val < other_val


class ElideLeftDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        painter.save()
        value = index.data(Qt.DisplayRole)
        textcolor = index.model().data(index, Qt.TextColorRole)
        pen = QPen(option.palette.text().color() if textcolor is None else textcolor.color())
        painter.setPen(pen)
        if int(option.state & QStyle.State_Selected) > 0:
            painter.fillRect(option.rect, QColor(20, 100, 160))
        else:
            backgroundColor = index.model().data(index, Qt.BackgroundColorRole)
            if backgroundColor:
                painter.fillRect(option.rect, backgroundColor.color())
        if index.column() == 0:
            painter.drawText(
                option.rect, Qt.AlignLeft | Qt.AlignVCenter,
                option.fontMetrics.elidedText(str(value), Qt.ElideLeft, option.rect.width()),
            )
        else:
            painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter, str(value))
        painter.restore()
