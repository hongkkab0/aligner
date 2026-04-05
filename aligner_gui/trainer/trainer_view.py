from __future__ import annotations

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from aligner_gui.ui.trainer_widget import Ui_trainer_widget
import logging
from aligner_gui.viewmodels.trainer_viewmodel import TrainerViewModel
from aligner_gui.shared import const
from aligner_gui.shared import gui_util
from aligner_gui.shared.graph_widget import GraphWidget
from aligner_engine.project_settings import ProjectSettings
from aligner_engine.summary import TrainSummary, ResultSummary
import numpy
from typing import Callable

TRAIN_LOGGER = logging.getLogger("aligner.trainer")


class TrainerView(QWidget, Ui_trainer_widget):
    COLOR_TRAINING = """QProgressBar::chunk { background: red; }"""
    COLOR_VALIDATION = """QProgressBar::chunk { background: green; }"""
    DEVICE_BAR_STYLE = """
    QProgressBar {
        background-color: rgb(28, 32, 38);
        border: 1px solid rgb(64, 74, 86);
        border-radius: 6px;
        padding: 1px;
    }
    QProgressBar::chunk {
        background-color: rgb(60, 170, 110);
        border-radius: 4px;
        margin: 1px;
    }
    """

    def __init__(self, main_window, session, tester_reload_callback: Callable[[], None] | None = None):
        super().__init__()
        self.setupUi(self)
        from aligner_gui.main_window import MainWindow
        self._main_window: MainWindow = main_window
        self._worker = session
        self._tester_reload_callback = tester_reload_callback

        self._canvas_graph = GraphWidget(hide_legend=True)
        self._canvas_graph_metric = GraphWidget(hide_legend=True, limit_range=(0, 1))
        self.layout_visualization.addWidget(self._canvas_graph)
        self.layout_visualization.addWidget(self._canvas_graph_metric)
        self.spin_resize.lineEdit().setReadOnly(True)
        self.spin_batch_size.setMaximum(256)
        self.spin_batch_size.setSingleStep(4)
        self.btn_train.setText("Train")
        self.check_resume = QCheckBox("Resume Last")
        self.check_resume.setToolTip("Continue training from the latest saved checkpoint.")
        self.horizontalLayout_4.addWidget(self.check_resume)

        self.device_panel = QFrame()
        self.device_panel.setMinimumWidth(120)
        self.device_panel.setMaximumWidth(160)
        self.device_panel.setFrameShape(QFrame.StyledPanel)
        self.device_panel.setFrameShadow(QFrame.Raised)
        self.device_panel_layout = QVBoxLayout(self.device_panel)
        self.device_panel_layout.setContentsMargins(8, 8, 8, 8)
        self.device_panel_layout.setSpacing(6)
        self.device_panel_layout.setAlignment(Qt.AlignTop)

        self.label_device_title = QLabel("GPU Mem")
        self.label_device_title.setAlignment(Qt.AlignCenter)
        self.device_panel_layout.addWidget(self.label_device_title)

        self.progress_device_usage = QProgressBar()
        self.progress_device_usage.setOrientation(Qt.Vertical)
        self.progress_device_usage.setFixedSize(20, 118)
        self.progress_device_usage.setRange(0, 100)
        self.progress_device_usage.setTextVisible(False)
        self.progress_device_usage.setInvertedAppearance(False)
        self.progress_device_usage.setStyleSheet(self.DEVICE_BAR_STYLE)
        self.progress_device_usage.hide()
        self.device_panel_layout.addWidget(self.progress_device_usage, alignment=Qt.AlignHCenter)

        self.label_device_percent = QLabel("")
        self.label_device_percent.setAlignment(Qt.AlignCenter)
        self.label_device_percent.setStyleSheet("font-weight: 600; color: rgb(220, 226, 232);")
        self.device_panel_layout.addWidget(self.label_device_percent)

        self.label_runtime_info = QLabel("VRAM --/--")
        self.label_runtime_info.setAlignment(Qt.AlignCenter)
        self.label_runtime_info.setStyleSheet("color: rgb(170, 180, 190);")
        self.label_runtime_info.setWordWrap(False)
        self.device_panel_layout.addWidget(self.label_runtime_info)
        self.device_panel_layout.addStretch(1)

        self.layout_visualization.addWidget(self.device_panel)

        # for training
        self._busy_indicator = QtGui.QMovie("aligner_gui\\icons\\essential\\ajax-loader_indicator_big_white.gif")
        self.lbl_train_indicator.setMovie(self._busy_indicator)
        self._busy_indicator.start()
        self.lbl_train_indicator.hide()
        self.viewmodel = TrainerViewModel(self, session, tester_reload_callback=tester_reload_callback)
        self.viewmodel.status_message_requested.connect(self._main_window.statusBar().showMessage)
        self.viewmodel.initialize()

    def _init_model_profile_ui(self):
        self.lbl_model_profile = QLabel("Model")
        self.lbl_model_profile.setMinimumSize(100, 0)
        self.combo_model_profile = QComboBox()
        self.combo_model_profile.setMinimumSize(180, 0)
        self.combo_model_profile.setToolTip(
            "Choose the detector profile to train and export.\n"
            "This is stored in the project config so different projects can use different model families."
        )
        self.horizontalLayout_4.insertWidget(0, self.combo_model_profile)
        self.horizontalLayout_4.insertWidget(0, self.lbl_model_profile)

        settings = self._get_settings()
        current_profile = settings.model_profile
        for profile in self._worker.get_model_profiles():
            self.combo_model_profile.addItem(profile.label, profile.id)
            idx = self.combo_model_profile.count() - 1
            self.combo_model_profile.setItemData(idx, profile.description, Qt.ToolTipRole)

        selected_index = self.combo_model_profile.findData(current_profile)
        if selected_index < 0:
            selected_index = self.combo_model_profile.findData("rotated_rtmdet_s")
        if selected_index >= 0:
            self.combo_model_profile.setCurrentIndex(selected_index)
            selected_profile_id = self.combo_model_profile.itemData(selected_index)
            if settings.model_profile != selected_profile_id:
                settings.model_profile = selected_profile_id
                self._save_settings(settings)
        self.combo_model_profile.setEnabled(False)
        self.combo_model_profile.setToolTip("Model profile switching is temporarily disabled until the corresponding pretrained weights are bundled.")
        self.lbl_model_profile.hide()
        self.combo_model_profile.hide()

    def _get_settings(self) -> ProjectSettings:
        return self.viewmodel.get_settings()

    def _save_settings(self, settings: ProjectSettings):
        self.viewmodel.save_settings(settings)

    def _clicked_check_hflip(self):
        self.viewmodel.clicked_check_hflip()

    def _clicked_check_vflip(self):
        self.viewmodel.clicked_check_vflip()

    def _clicked_check_no_rotation(self):
        self.viewmodel.clicked_check_no_rotation()

    def _clicked_check_include_empty(self):
        self.viewmodel.clicked_check_include_empty()

    def _changed_spin_resize(self):
        self.viewmodel.changed_spin_resize()

    def _changed_max_epochs(self):
        self.viewmodel.changed_max_epochs()

    def _changed_spin_batch_size(self):
        self.viewmodel.changed_spin_batch_size()

    def _changed_model_profile(self):
        self.viewmodel.changed_model_profile()

    def _get_torch(self):
        return self.viewmodel.get_torch()

    def _prepare_training_assets(self) -> bool:
        return self.viewmodel.prepare_training_assets()

    def _refresh_resume_ui(self):
        self.viewmodel.refresh_resume_ui()

    def _refresh_existing_training_monitor(self, last_epoch: int):
        self.viewmodel.refresh_existing_training_monitor(last_epoch)

    def _start_training(self):
        self.viewmodel.start_training()

    def _stop_training(self, reason: str):
        self.viewmodel.stop_training(reason)

    def _on_start_training(self, resume_training: bool):
        self.viewmodel.on_start_training(resume_training)

    def _on_stop_training(self):
        self.viewmodel.on_stop_training()

    def _update_training_status(self, message: str):
        self.viewmodel.update_training_status(message)

    def _clicked_btn_train(self):
        self.viewmodel.handle_train_button_clicked()

    def close(self):
        self.viewmodel.close()
        return super().close()

    def _refresh_training_time(self, cur_epoch):
        self.viewmodel.refresh_training_time(cur_epoch)

    def _refresh_training_time_live(self, phase_type: str, iter_idx: int, iter_len: int):
        self.viewmodel.refresh_training_time_live(phase_type, iter_idx, iter_len)

    def _sample_gpu_memory_stats(self):
        return self.viewmodel.sample_gpu_memory_stats()

    def _refresh_device_usage(self):
        self.viewmodel.refresh_device_usage()

    def _apply_gpu_snapshot(self, snapshot: GpuUsageSnapshot):
        self.viewmodel.apply_gpu_snapshot(snapshot)

    def _handle_gpu_monitor_unavailable(self, _: str):
        self.viewmodel.handle_gpu_monitor_unavailable(_)

    def _apply_gpu_memory_fallback(self):
        self.viewmodel.apply_gpu_memory_fallback()

    def _update_epoch_th_train(self, current_epoch: int, current_ckpt_path: str):
        self.viewmodel.update_epoch(current_epoch, current_ckpt_path)

    def _update_iter_th_train(self, phase_type: str, iter_idx: int, iter_len: int):
        self.viewmodel.update_iter(phase_type, iter_idx, iter_len)

    def show(self):
        super().show()
        self._refresh_resume_ui()

        if len(self._worker.get_train_summary().tr_by_epoch.keys()) == 0:
            return

        try:
            last_epoch = list(self._worker.get_train_summary().tr_by_epoch.keys())[-1]
            self._refresh_training_chart(self._worker.get_train_summary(), self._worker.get_train_result_summary(),
                                         self._worker.get_valid_result_summary())
            self._refresh_training_history_table(self._worker.get_train_summary(),
                                                 self._worker.get_train_result_summary(),
                                                 self._worker.get_valid_result_summary())
            self._refresh_validation_table(last_epoch, self._worker.get_valid_result_summary())
        except Exception as e:
            pass


    def _init_training_chart(self):
        self._canvas_graph.reset()
        self._canvas_graph_metric.reset()
        self._canvas_graph.setName('Train Loss')
        self._canvas_graph.setBottomName('Epoch')
        self._canvas_graph.setLine('Loss', [], QtGui.QColor(205, 92, 92), 2)

        self._canvas_graph_metric.setName(f'Validation {self._worker.metric_name}')
        self._canvas_graph_metric.setBottomName('Epoch')
        self._canvas_graph_metric.setLine(self._worker.metric_name, [], QtGui.QColor(92, 205, 92), 2)

    def _init_training_history(self):
        self.table_training_history.clear()
        h_names = ['Train\nLoss', 'Valid\n%s' % self._worker.metric_name,
                   'Corner\nError',
                   'Corner\nX',
                   'Corner\nY',
                   'Center\nError',
                   'Center\nX',
                   'Center\nY',
                   'Longside\nError',
                   'Shortside\nError', 'Update']

        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(1)

    def _init_validation_table(self):
        self.table_validation.clear()

    def _refresh_training_chart(self, worker_train_summary: TrainSummary,
                                worker_train_result_summary: ResultSummary,
                                worker_valid_result_summary: ResultSummary):
        tr_loss = []
        va_metric = []
        for epoch in worker_train_summary.tr_by_epoch:
            tr_loss.append(worker_train_summary.tr_by_epoch[epoch]['loss'])

            if epoch in worker_train_summary.va_by_epoch:
                if epoch in worker_valid_result_summary.map:
                    va_metric.append(worker_valid_result_summary.get_metric(epoch, self._worker.metric_name))
                else:
                    if len(va_metric) > 0:
                        va_metric.append(va_metric[-1])
                    else:
                        va_metric.append(0)
            else:
                va_metric.append(numpy.nan)

        self._canvas_graph.setName('Train Loss')
        self._canvas_graph.setLine('Loss', tr_loss, QtGui.QColor(205, 92, 92), 2)

        self._canvas_graph_metric.setName(f'Validation {self._worker.metric_name}')
        self._canvas_graph_metric.setLine(self._worker.metric_name, va_metric, QtGui.QColor(92, 205, 92), 2)

    def _refresh_training_history_table(self, worker_train_summary: TrainSummary,
                                        worker_train_result_summary: ResultSummary,
                                        worker_valid_result_summary: ResultSummary):
        self.table_training_history.clear()

        h_names = ['Train\nLoss', 'Valid\n%s' % self._worker.metric_name,
                   'Corner\nError',
                   'Corner\nX',
                   'Corner\nY',
                   'Center\nError',
                   'Center\nX',
                   'Center\nY',
                   'Longside\nError',
                   'Shortside\nError', 'Update']
        self.table_training_history.setColumnCount(len(h_names))
        self.table_training_history.setHorizontalHeaderLabels(h_names)
        self.table_training_history.setRowCount(len(worker_train_summary.tr_by_epoch))

        header_epoch_map = {}

        for idx, (ep, info) in enumerate(worker_train_summary.tr_by_epoch.items()):
            self.table_training_history.setVerticalHeaderItem(idx, QTableWidgetItem(str(ep)))
            self.table_training_history.setItem(idx, 0, QTableWidgetItem('%.4lf' % info['loss']))
            header_epoch_map[ep] = idx

        for ep, info in worker_train_summary.va_by_epoch.items():
            idx = header_epoch_map[ep]
            self.table_training_history.setItem(idx, 1, QTableWidgetItem(
                '%.4lf' % worker_valid_result_summary.get_metric(ep, self._worker.metric_name)))

            mPE = worker_valid_result_summary.get_metric(ep, 'mPE')
            corner_error = f'{mPE["corner_error"]:.1f}'
            self.table_training_history.setItem(idx, 2, QTableWidgetItem(corner_error))

            corner_dx = f'{mPE["corner_dx"]:.1f}'
            self.table_training_history.setItem(idx, 3, QTableWidgetItem(corner_dx))

            corner_dy = f'{mPE["corner_dy"]:.1f}'
            self.table_training_history.setItem(idx, 4, QTableWidgetItem(corner_dy))

            center_error = f'{mPE["center_error"]:.1f}'
            self.table_training_history.setItem(idx, 5, QTableWidgetItem(center_error))

            center_dx = f'{mPE["center_dx"]:.1f}'
            self.table_training_history.setItem(idx, 6, QTableWidgetItem(center_dx))

            center_dy = f'{mPE["center_dy"]:.1f}'
            self.table_training_history.setItem(idx, 7, QTableWidgetItem(center_dy))

            longside = f'{mPE["longside"]:.1f}'
            self.table_training_history.setItem(idx, 8, QTableWidgetItem(longside))

            shortside = f'{mPE["shortside"]:.1f}'
            self.table_training_history.setItem(idx, 9, QTableWidgetItem(shortside))

        for updated_epoch in worker_train_summary.update_model_epoch:
            idx = header_epoch_map[updated_epoch]
            self.table_training_history.setItem(idx, 10, QTableWidgetItem('O'))

        self.table_training_history.resizeColumnsToContents()
        self.table_training_history.resizeRowsToContents()
        self.table_training_history.scrollToBottom()

    def _refresh_validation_table(self, epoch: int, worker_valid_result_summary: ResultSummary):
        self.table_validation.clear()
        self.table_validation.clearSpans()
        n_class = len(worker_valid_result_summary.class_name)
        if n_class <= 0:
            return

        h_class_names = [worker_valid_result_summary.class_name[i] for i in range(n_class)]
        self.table_validation.setColumnCount(len(h_class_names))
        self.table_validation.setHorizontalHeaderLabels(h_class_names)
        v_class_names = ['#', 'AP', 'Epoch', self._worker.metric_name, 'Corner Error', 'Corner X', 'Corner Y',
                         'Center Error', 'Center X', 'Center Y', 'Width Error', 'Height Error']
        self.table_validation.setRowCount(len(v_class_names))
        self.table_validation.setVerticalHeaderLabels(v_class_names)

        def format_metric(value):
            if value is None:
                return '-'
            try:
                if numpy.isnan(value):
                    return '-'
            except TypeError:
                pass
            return f'{value:.1f}'

        if epoch in worker_valid_result_summary.aps:
            aps = worker_valid_result_summary.aps[epoch]
            if aps is not None:
                for c in range(n_class):
                    self.table_validation.setItem(0, c, QTableWidgetItem(str(int(aps[c][1]))))  # num gts
                    self.table_validation.item(0, c).setTextAlignment(Qt.AlignCenter)
                    self.table_validation.item(0, c).setBackground(QtGui.QColor(144, 164, 174))

                    self.table_validation.setItem(1, c, QTableWidgetItem('%.2lf' % aps[c][0]))  # ap
                    self.table_validation.item(1, c).setTextAlignment(Qt.AlignCenter)
                    self.table_validation.item(1, c).setBackground(QtGui.QColor(144, 164, 174))

        def set_overall_row(row_no: int, text: str):
            self.table_validation.setSpan(row_no, 0, 1, n_class)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QtGui.QColor(65, 78, 92))
            self.table_validation.setItem(row_no, 0, item)

        set_overall_row(2, str(epoch))

        if epoch in worker_valid_result_summary.map:
            map = worker_valid_result_summary.map[epoch]
            set_overall_row(3, '%.4lf' % map)

        if epoch in worker_valid_result_summary.mpe:
            mpe = worker_valid_result_summary.get_metric(epoch, 'mPE')
            class_metrics = worker_valid_result_summary.mpe_by_class.get(epoch, {})
            if class_metrics:
                row_key_map = {
                    4: 'corner_error',
                    5: 'corner_dx',
                    6: 'corner_dy',
                    7: 'center_error',
                    8: 'center_dx',
                    9: 'center_dy',
                    10: 'longside',
                    11: 'shortside',
                }
                for row_no, metric_key in row_key_map.items():
                    for c in range(n_class):
                        value = class_metrics.get(c, {}).get(metric_key)
                        item = QTableWidgetItem(format_metric(value))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table_validation.setItem(row_no, c, item)
            else:
                set_overall_row(4, f'{mpe["corner_error"]:.1f}')
                set_overall_row(5, f'{mpe["corner_dx"]:.1f}')
                set_overall_row(6, f'{mpe["corner_dy"]:.1f}')
                set_overall_row(7, f'{mpe["center_error"]:.1f}')
                set_overall_row(8, f'{mpe["center_dx"]:.1f}')
                set_overall_row(9, f'{mpe["center_dy"]:.1f}')
                set_overall_row(10, f'{mpe["longside"]:.1f}')
                set_overall_row(11, f'{mpe["shortside"]:.1f}')

        self.table_validation.resizeColumnsToContents()
        self.table_validation.resizeRowsToContents()

    def _clicked_table_training_history(self):
        epoch = self.table_training_history.currentRow() + 1
        self._refresh_validation_table(epoch, self._worker.get_valid_result_summary())

