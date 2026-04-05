from __future__ import annotations

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QMovie, QPen, QPainter, QFont, QCursor, QGuiApplication
from PyQt5 import QtGui
import subprocess

from aligner_gui.viewmodels.tester_viewmodel import TesterViewModel
from aligner_gui.ui.tester_widget import Ui_tester_widget
from aligner_gui.tester.preview_renderer import TesterPreviewRenderer
import logging
import traceback
from aligner_gui.shared import const
from aligner_gui.shared import gui_util
from aligner_gui.shared.image_panel import ImagePanel
import json
from copy import deepcopy
import os
from datetime import datetime
from aligner_gui.shared import io_util
import numpy as np

TEST_LOGGER = logging.getLogger("aligner.tester")


class TesterView(QWidget, Ui_tester_widget):
    COLOR_INFERENCE = """QProgressBar::chunk { background: yellow; }"""

    def __init__(self, session, is_new: bool):
        super().__init__()
        self.setupUi(self)
        self._is_new = is_new

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\plus(white).png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_add.setIcon(icon)

        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\folder-10(white).png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_add_folder.setIcon(icon)

        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\garbage-2(white).png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_delete.setIcon(icon)

        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\075-reload.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_reset.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\083-share.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_export_csv.setIcon(icon)
        icon.addPixmap(QtGui.QPixmap("aligner_gui\\icons\\essential\\save(white).png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.btn_save_image.setIcon(icon)

        self.btn_add.clicked.connect(self._clicked_btn_add)
        self.btn_add_folder.clicked.connect(self._clicked_btn_add_folder)
        self.btn_delete.clicked.connect(self._clicked_btn_delete)
        self.btn_reset.clicked.connect(self._clicked_btn_reset)
        self.btn_export_csv.clicked.connect(self._clicked_btn_export_csv)
        self.btn_save_image.clicked.connect(self._clicked_btn_save_image)

        # test setting
        self.btn_test.clicked.connect(self._clicked_btn_test)

        # for testing
        self._busy_indicator = QMovie("aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif")
        self.lbl_test_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_test_indicator.hide()

        # image panel
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

        # table
        self._delegate_table_test_detail = ElideLeftDelegate()
        self.table_test_detail.setItemDelegate(self._delegate_table_test_detail)
        self.table_test_detail.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_test_detail.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_test_detail.itemSelectionChanged.connect(self._item_selected_changed_table_detail)
        self.table_test_detail.cellPressed.connect(self._cell_pressed_table_detail)
        self.table_test_detail.horizontalHeader().sectionClicked.connect(self._section_clicked_table_detail)
        self._src_path_clicked_of_table_detail: str = ""
        self.table_test_detail_sort_col: int = -1

        self._delegate_table_file_list = ElideLeftDelegate()
        self.table_file_list.setItemDelegate(self._delegate_table_file_list)
        self.table_file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_file_list.itemSelectionChanged.connect(self._item_selected_changed_table_file_list)

        # initialize file list infos
        self._classes = []
        self._class_index = {}
        self._class_name = {}
        self._last_preview_key = None

        self.viewmodel = TesterViewModel(session, parent=self)
        self.viewmodel.testing_started.connect(self._on_testing_started)
        self.viewmodel.testing_stopped.connect(self._on_testing_stopped)
        self.viewmodel.iter_progress_updated.connect(self._on_iter_progress_updated)
        self.viewmodel.results_updated.connect(self._on_results_updated)
        self.viewmodel.test_blocked.connect(
            lambda title, msg: gui_util.get_message_box(self, title, msg)
        )

        self._preview_renderer = TesterPreviewRenderer(self._get_cv2, self.viewmodel.get_project_settings)

        if not self._is_new:
            self.reload_file_list()

    def _get_torch(self):
        import torch

        return torch

    def _get_cv2(self):
        import cv2

        return cv2

    def _clear_preview_cache(self):
        self._preview_renderer.clear()
        self._last_preview_key = None

    def _refresh_current_preview(self):
        self._last_preview_key = None
        indexes = self.table_test_detail.selectedIndexes()
        if len(indexes) > 0:
            idx = indexes[0].row()
            img_path = self.table_test_detail.item(idx, 0).text()
            self._change_selected_image_table_detail(img_path)
            return

        indexes = self.table_file_list.selectedIndexes()
        if len(indexes) > 0:
            idx = indexes[0].row()
            img_path = self.table_file_list.item(idx, 0).text()
            self._change_selected_image_table_file_list(img_path)

    def _append_files(self, paths: list) -> None:
        self.viewmodel.append_files(paths)

    def _clicked_btn_test(self) -> None:
        self.viewmodel.handle_test_button_clicked(self.btn_test.isChecked())

    def close(self) -> bool:
        self.viewmodel.close()
        return super().close()

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

    def _on_testing_stopped(self, _reason: str) -> None:
        self.btn_test.setChecked(False)
        self.btn_test.setEnabled(True)
        self.progress_iter.setMaximum(1)
        self.progress_iter.setValue(0)
        self.progress_iter.setStyleSheet(gui_util.get_dark_style())
        self.lbl_test_indicator.hide()

    def _on_iter_progress_updated(self, idx: int, total: int) -> None:
        self.progress_iter.setStyleSheet(self.COLOR_INFERENCE)
        self.progress_iter.setMaximum(total)
        self.progress_iter.setValue(idx + 1)
        self.btn_test.setEnabled(True)

    def _on_results_updated(self) -> None:
        self._worker_test_result_summary = self.viewmodel.get_test_result_summary()
        self._clear_preview_cache()
        self._refresh_test_detail_table()
        self._refresh_test_time()

    def _refresh_test_detail_table(self):
        selected_img_path = None
        selected_indexes = self.table_test_detail.selectedIndexes()
        if len(selected_indexes) > 0:
            selected_item = self.table_test_detail.item(selected_indexes[0].row(), 0)
            if selected_item is not None:
                selected_img_path = selected_item.text()

        table_blocker = QSignalBlocker(self.table_test_detail)
        self.table_test_detail.setUpdatesEnabled(False)
        self.table_test_detail.clear()
        # self.table_test_detail.setColumnCount(0)
        self.table_test_detail.setSortingEnabled(False)
        # max_roi = self.workerWidget.worker.infer_config['max_detection']
        max_roi = 1
        data_results = self._worker_test_result_summary.data_result
        for img_path, data_result in data_results.items():
            data_result_len = len(data_result)
            if data_result_len > max_roi:
                max_roi = data_result_len

        column_names = ['Path', 'Corner\nError', 'Corner\nX', 'Corner\nY',
                        'Center\nError', 'Center\nX', 'Center\nY', 'Longside\nError', 'Shortside\nError', '# ROI']
        for i in range(max_roi):
            column_names.append('R%d' % i)
        self.table_test_detail.setColumnCount(len(column_names))
        self.table_test_detail.setHorizontalHeaderLabels(column_names)
        self.table_test_detail.setColumnWidth(0, 100)
        for i in range(1, len(column_names)):
            self.table_test_detail.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table_test_detail.setRowCount(len(self._worker_test_result_summary.data_result))

        data_results = self._worker_test_result_summary.data_result
        for idx, (img_path, data_result) in enumerate(data_results.items()):
            self.table_test_detail.setItem(idx, 0, QTableWidgetItem(img_path))

            diffs = [data_result[str(i)]['corner_error'] for i in range(len(data_result)) if
                     'corner_error' in data_result[str(i)].keys()]
            corner_error = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 1, QTableWidgetItemNum(corner_error))

            diffs = [data_result[str(i)]['corner_dx'] for i in range(len(data_result)) if
                     'corner_dx' in data_result[str(i)].keys()]
            corner_dx = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 2, QTableWidgetItemNum(corner_dx))

            diffs = [data_result[str(i)]['corner_dy'] for i in range(len(data_result)) if
                     'corner_dy' in data_result[str(i)].keys()]
            corner_dy = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 3, QTableWidgetItemNum(corner_dy))

            diffs = [data_result[str(i)]['center_error'] for i in range(len(data_result)) if
                     'center_error' in data_result[str(i)].keys()]
            center_error = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 4, QTableWidgetItemNum(center_error))

            diffs = [data_result[str(i)]['center_dx'] for i in range(len(data_result)) if
                     'center_dx' in data_result[str(i)].keys()]
            center_dx = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 5, QTableWidgetItemNum(center_dx))

            diffs = [data_result[str(i)]['center_dy'] for i in range(len(data_result)) if
                     'center_dy' in data_result[str(i)].keys()]
            center_dy = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 6, QTableWidgetItemNum(center_dy))

            diffs = [data_result[str(i)]['longside'] for i in range(len(data_result)) if
                     'longside' in data_result[str(i)].keys()]
            longside = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 7, QTableWidgetItemNum(longside))

            diffs = [data_result[str(i)]['shortside'] for i in range(len(data_result)) if
                     'shortside' in data_result[str(i)].keys()]
            shortside = '%.2lf' % (np.asarray(diffs).mean()) if len(diffs) > 0 else '-'
            self.table_test_detail.setItem(idx, 8, QTableWidgetItemNum(shortside))

            self.table_test_detail.setItem(idx, 9, QTableWidgetItemNum(str(len(data_result))))

            for i in range(len(data_result)):
                self.table_test_detail.setItem(idx, 10 + i,
                                               QTableWidgetItem('%s (%.2lf)' % (
                                                   (data_result[str(i)]["class_name"]), (data_result[str(i)]["conf"]))))

        if selected_img_path is not None:
            for row in range(self.table_test_detail.rowCount()):
                item = self.table_test_detail.item(row, 0)
                if item is not None and item.text() == selected_img_path:
                    self.table_test_detail.selectRow(row)
                    break
        self.table_test_detail.setUpdatesEnabled(True)
        del table_blocker

    def _refresh_test_time(self) -> None:
        self.label_time.setText("%.3fsec/image" % self.viewmodel.get_mean_test_time())

    def _change_selected_image_table_file_list(self, img_path):
        preview_key = ("file", img_path)
        if self._last_preview_key == preview_key:
            return

        qpixmap = self._preview_renderer.render_file_preview(img_path)
        if qpixmap is None:
            return
        self.image_panel.setImage(qpixmap)
        self._last_preview_key = preview_key

    def _change_selected_image_table_detail(self, img_path):
        preview_key = ("detail", img_path, self.check_show_gt.isChecked(), self.check_show_prediction.isChecked())
        if self._last_preview_key == preview_key:
            return
        qpixmap = self._preview_renderer.render_detail_preview(
            img_path,
            self._worker_test_result_summary,
            self._class_index,
            self.check_show_gt.isChecked(),
            self.check_show_prediction.isChecked(),
        )
        if qpixmap is None:
            return
        self.image_panel.setImage(qpixmap)
        self._last_preview_key = preview_key

    @pyqtSlot(int, int)
    def _cell_pressed_table_detail(self, idx, col):
        if self.table_test_detail.rowCount() < 1 or len(self.table_test_detail.selectedItems()) != 1:
            return
        if (col == 0) and QApplication.mouseButtons() & Qt.RightButton:
            self._src_path_clicked_of_table_detail = self.table_test_detail.item(idx, col).text().replace("\\", "/")

            self.menuValues = QMenu(self)
            self.signalMapper = QSignalMapper(self)

            open_src_path_action = QAction("Open Source File", self)
            open_src_path_action.triggered.connect(self.on_open_src_path_triggered)
            self.menuValues.addAction(open_src_path_action)

            open_src_folder_action = QAction("Open Source Folder", self)
            open_src_folder_action.triggered.connect(self.on_open_src_folder_triggered)
            self.menuValues.addAction(open_src_folder_action)

            self.menuValues.exec_(QPoint(QCursor.pos().x() + 2, QCursor.pos().y() + 2))
            return

    def on_open_src_path_triggered(self):
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            os.startfile(self._src_path_clicked_of_table_detail)
        except Exception as e:
            print(e)
        QGuiApplication.restoreOverrideCursor()

    def on_open_src_folder_triggered(self):
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            subprocess.Popen(
                r'explorer /select,"' + os.path.realpath(self.table_test_detail.selectedItems()[0].text()) + '"')
        except Exception as e:
            print(e)
        QGuiApplication.restoreOverrideCursor()

    def _item_selected_changed_table_detail(self):
        indexes = self.table_test_detail.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0] if isinstance(indexes, list) else indexes
            idx = index.row()
            img_path = self.table_test_detail.item(idx, 0).text()
            self._change_selected_image_table_detail(img_path)

    def _item_selected_changed_table_file_list(self):
        indexes = self.table_file_list.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0] if isinstance(indexes, list) else indexes
            idx = index.row()
            img_path = self.table_file_list.item(idx, 0).text()
            self._change_selected_image_table_file_list(img_path)

    @pyqtSlot(int)
    def _section_clicked_table_detail(self, col):
        if col in [1, 2, 3, 4, 5, 6, 7, 8, 9]:  # for sorting roi count
            self.table_test_detail.setSortingEnabled(False)
            self.table_test_detail_sort_col = col
            self.menuValues = QMenu(self)
            self.signalMapper = QSignalMapper(self)

            sort_ascending = QAction("Ascending", self)
            sort_ascending.triggered.connect(self._on_sort_ascending)
            sort_descending = QAction("Descending", self)
            sort_descending.triggered.connect(self._on_sort_descending)
            self.menuValues.addAction(sort_ascending)
            self.menuValues.addAction(sort_descending)

            header_pos = self.table_test_detail.mapToGlobal(self.table_test_detail.horizontalHeader().pos())

            pos_y = header_pos.y() + self.table_test_detail.horizontalHeader().height()
            pos_x = header_pos.x() + self.table_test_detail.horizontalHeader().sectionViewportPosition(
                self.table_test_detail_sort_col)

            self.menuValues.exec_(QPoint(pos_x, pos_y))
        else:
            self.table_test_detail.setSortingEnabled(False)

    def _on_sort_ascending(self):
        self.table_test_detail.setSortingEnabled(True)
        self.table_test_detail.sortByColumn(self.table_test_detail_sort_col, Qt.AscendingOrder)

    def _on_sort_descending(self):
        self.table_test_detail.setSortingEnabled(True)
        self.table_test_detail.sortByColumn(self.table_test_detail_sort_col, Qt.DescendingOrder)

    def _clicked_btn_add(self) -> None:
        file_paths = gui_util.get_files_from_dialog(
            self, is_open=True, ext_list=gui_util.SUPPORTED_IMAGE_FORMATS_WITHOUT_DOT
        )
        if not file_paths:
            return
        self.viewmodel.append_files(file_paths)
        self._clear_preview_cache()
        self._refresh_table_file_list()

    def _scan_all_images(self, folder_path):
        extensions = gui_util.SUPPORTED_IMAGE_FORMATS
        images = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relatvie_path = os.path.join(root, file)
                    path = os.path.abspath(relatvie_path)
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def _clicked_btn_add_folder(self) -> None:
        dir_path = gui_util.get_open_dir_from_dialog(self, "Choose a directory to test images")
        if dir_path == "":
            return
        self.viewmodel.append_files(self._scan_all_images(dir_path))
        self._clear_preview_cache()
        self._refresh_table_file_list()

    def _clicked_btn_delete(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.table_file_list.selectionModel().selectedIndexes()
             if idx.column() == 0},
            reverse=True,
        )
        if rows:
            self.viewmodel.remove_files_at_rows(rows)
            self._clear_preview_cache()
            self._refresh_table_file_list()

    def _clicked_btn_save_image(self):
        extensions = ["bmp", "png", "jpg"]
        cur_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_file_name = "Image__%s.jpg" % cur_time_str
        file_path = gui_util.get_file_from_dialog(self, is_open=False, ext_list=extensions, save_file_name=save_file_name)
        if file_path == "":
            return
        self.image_panel.saveImage(file_path)

    def _clicked_btn_reset(self):
        self._clear_preview_cache()
        self.reload_file_list()

    def _clicked_btn_export_csv(self):
        if self.table_test_detail.rowCount() <= 0:
            return
        save_path = gui_util.get_path_from_dialog(self, 'Export test result')
        cur_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        file_name = '[dice_aligner_test_result] %s.csv' % (cur_time_str)
        file_name = io_util.join_path(save_path, file_name)
        gui_util.table2csv(self.table_test_detail, file_name)
        gui_util.get_message_box(self, "Export test result", "Test result is saved to %s." % (file_name))

    def reload_file_list(self) -> None:
        try:
            dataset_summary_path = self.viewmodel.get_dataset_summary_path()
            with open(dataset_summary_path, "r", encoding="utf-8") as f:
                self._dataset_summary = json.load(f)
        except Exception as e:
            TEST_LOGGER.info(e)
            return

        self._classes = [c["name"] for c in self._dataset_summary["class_summary"]["classes"]]
        self._class_index = {v: idx for idx, v in enumerate(self._classes)}
        self._class_name = {idx: v for idx, v in enumerate(self._classes)}
        self.viewmodel.reset_file_list([data["img_path"] for data in self._dataset_summary["data_summary"]])
        self._clear_preview_cache()
        self._refresh_table_file_list()

    def _refresh_table_file_list(self) -> None:
        file_list = self.viewmodel.get_file_list()
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
            label = "O" if self._exist_label_file_of_img_path(img_path) else "X"
            self.table_file_list.setItem(idx, 1, QTableWidgetItem(label))
            if img_path in selected_paths:
                self.table_file_list.selectRow(idx)
        self.table_file_list.setUpdatesEnabled(True)
        del table_blocker

    def _get_label_path_from_img_path(self, img_path):
        return os.path.splitext(img_path)[0] + ".json"

    def _exist_label_file_of_img_path(self, img_path):
        label_path = self._get_label_path_from_img_path(img_path)
        return os.path.exists(label_path)

class QTableWidgetItemNum(QTableWidgetItem):
    def __lt__(self, other):  # self < other
        self_val = -1.0 if self.text() == '-' else float(self.text())
        other_val = -1.0 if other.text() == '-' else float(other.text())
        return self_val < other_val


class ElideLeftDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def paint(self, painter, option, index):
        painter.save()
        value = index.data(Qt.DisplayRole)
        textcolor = index.model().data(index, Qt.TextColorRole)
        pen = QPen(option.palette.text().color() if textcolor == None else textcolor.color())
        painter.setPen(pen)
        if int(option.state & QStyle.State_Selected) > 0:
            painter.fillRect(option.rect, QColor(20, 100, 160))
        else:
            backgroundColor = index.model().data(index, Qt.BackgroundColorRole)
            if backgroundColor:
                painter.fillRect(option.rect, backgroundColor.color())
        if index.column() == 0:
            painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter,
                             option.fontMetrics.elidedText(str(value), Qt.ElideLeft, option.rect.width()))
        else:
            painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter, str(value))
        painter.restore()

