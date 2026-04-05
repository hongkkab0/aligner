from __future__ import annotations

import importlib
import logging
import os
from typing import Dict

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox

from aligner_gui import __appname__
from aligner_gui.utils import const
from aligner_gui.utils import gui_util
from aligner_gui.utils.log_manager import configure_project_logging
from aligner_gui.widgets.log_widget import LogWidget
from aligner_gui.widgets.progress_general_dialog import ProgressGeneralDialog
from aligner_gui.project.project_export_service import export_project_bundle, list_export_checkpoint_options
from aligner_gui.dialogs.project_open_dialog import ProjectOpenDialog
from aligner_gui.project.project_session import ProjectSession
from aligner_gui.ui.main_window import Ui_main_window
from aligner_gui.viewmodels.app_viewmodel import AppViewModel


class StartupWarmupThread(QtCore.QThread):
    sig_progress = QtCore.pyqtSignal(int, int, str)

    def __init__(self, steps):
        super().__init__()
        self._steps = list(steps)
        self.error_message = ""

    @staticmethod
    def _import_training_stack():
        import mmcv  # noqa: F401
        import torch  # noqa: F401
        from mmengine.config import Config  # noqa: F401
        from mmdet.registry import RUNNERS  # noqa: F401
        from mmrotate.registry import HOOKS  # noqa: F401

    @staticmethod
    def _import_inference_stack():
        import cv2  # noqa: F401
        from mmengine.config import Config  # noqa: F401
        from mmdet.apis import inference_detector, init_detector  # noqa: F401
        import aligner_engine.detector  # noqa: F401

    def run(self):
        total = len(self._steps)
        for idx, (message, func) in enumerate(self._steps, start=1):
            self.sig_progress.emit(idx, total, message)
            try:
                func()
            except Exception as exc:
                logging.exception("Startup warmup step failed: %s", message)
                self.error_message = f"{message} {exc}"
                break


class MainWindow(QMainWindow, Ui_main_window):
    TAB_LABELER = "labeler"
    TAB_TRAINER = "trainer"
    TAB_TESTER = "tester"

    def __init__(self, project_path, project_open_type):
        super().__init__()
        self.setupUi(self)

        self._app_status = const.APP_STATUS_IDLE
        self._tab_widgets: Dict[str, object] = {}
        self._loading_label = QLabel("Loading...")
        self._loading_label.setAlignment(QtCore.Qt.AlignCenter)

        self._project_path = ""
        self._project_open_type = ""
        self._is_new = False
        self.session: ProjectSession | None = None
        self._prewarm_queue = []
        self._prewarm_active = False
        self._background_warmup_threads = []

        self.app_vm = AppViewModel(self)
        self.app_vm.status_changed.connect(self._on_app_status_changed)

        self._init_ui()
        self._replace_project(project_path, project_open_type)
        self._on_app_status_changed(const.APP_STATUS_IDLE)

    def _init_ui(self):
        self.trainer_log_widget = LogWidget(prefixes=("aligner.trainer",))
        self.tester_log_widget = LogWidget(prefixes=("aligner.tester",))
        gui_util.update_layout(self.layout_log, self.trainer_log_widget)
        self.log_widget_container.hide()

        self.setWindowIcon(QtGui.QIcon("aligner_gui\\icons\\012-left-align-4.png"))
        self.action_labeler.setIcon(QIcon("aligner_gui\\icons\\essential\\bookmark(white).png"))
        self.action_trainer.setIcon(QIcon("aligner_gui\\icons\\essential\\002-book(white).png"))
        self.action_test.setIcon(QIcon("aligner_gui\\icons\\essential\\list(white).png"))
        self.action_export.setIcon(QIcon("aligner_gui\\icons\\essential\\078-folder-73(white).png"))

        self.action_labeler.triggered.connect(self.triggered_action_labeler)
        self.action_trainer.triggered.connect(self.triggered_action_trainer)
        self.action_test.triggered.connect(self.triggered_action_test)
        self.action_export.triggered.connect(self.triggered_action_export)
        self.action_new_project.triggered.connect(self.triggered_action_new_project)
        self.action_load_project.triggered.connect(self.triggered_action_load_project)
        self.statusBar().showMessage("Ready")

    def _replace_project(self, project_path, project_open_type):
        self._stop_background_warmups()
        self._clear_tab_widgets()
        if self.session is not None:
            self.session.close()

        self._project_path = project_path
        self._project_open_type = project_open_type
        self._is_new = project_open_type == const.PROJECT_OPEN_TYPE_NEW
        configure_project_logging(project_path)
        self.session = ProjectSession(project_path, is_new=self._is_new)
        self._prewarm_queue = []
        self._prewarm_active = False
        self.setWindowTitle(__appname__ + " - " + project_path)
        self.log_widget_container.hide()
        gui_util.update_layout(self.layout_main, self._loading_label)

    def _stop_background_warmups(self):
        for thread in list(self._background_warmup_threads):
            try:
                thread.wait(3000)
            except Exception:
                pass
        self._background_warmup_threads.clear()

    def _clear_tab_widgets(self):
        for widget in self._tab_widgets.values():
            try:
                widget.close()
            except Exception:
                pass
            widget.deleteLater()
        self._tab_widgets.clear()
        gui_util.hide_layout(self.layout_main)

    def _create_tab_widget(self, tab_name: str):
        assert self.session is not None

        if tab_name == self.TAB_LABELER:
            from aligner_gui.labeler.labeler_widget import LabelerWidget

            return LabelerWidget(self, self.session, self._project_path, is_new=self._is_new)
        if tab_name == self.TAB_TESTER:
            from aligner_gui.tester.tester_widget import TesterWidget

            return TesterWidget(self.session, is_new=self._is_new)
        if tab_name == self.TAB_TRAINER:
            from aligner_gui.trainer.trainer_widget import TrainerWidget

            return TrainerWidget(
                self.session,
                self.app_vm,
                tester_reload_callback=self._reload_tester_if_loaded,
            )
        raise NotImplementedError(tab_name)

    def _get_tab_widget(self, tab_name: str):
        widget = self._tab_widgets.get(tab_name)
        if widget is not None:
            return widget

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage(f"Loading {tab_name}...")
        QApplication.processEvents()
        try:
            widget = self._create_tab_widget(tab_name)
            self._tab_widgets[tab_name] = widget
            return widget
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Ready", 3000)

    def _show_tab(self, tab_name: str, show_log: bool):
        if show_log:
            self.log_widget_container.show()
            if tab_name == self.TAB_TRAINER:
                gui_util.update_layout(self.layout_log, self.trainer_log_widget)
            elif tab_name == self.TAB_TESTER:
                gui_util.update_layout(self.layout_log, self.tester_log_widget)
        else:
            self.log_widget_container.hide()

        widget = self._get_tab_widget(tab_name)
        gui_util.update_layout(self.layout_main, widget)

        self.action_labeler.setChecked(tab_name == self.TAB_LABELER)
        self.action_trainer.setChecked(tab_name == self.TAB_TRAINER)
        self.action_test.setChecked(tab_name == self.TAB_TESTER)
        self.setWindowTitle(__appname__ + " - " + self._project_path)

    def _reload_tester_if_loaded(self):
        tester_widget = self._tab_widgets.get(self.TAB_TESTER)
        if tester_widget is not None:
            tester_widget.reload_file_list()

    def _get_startup_warmup_steps(self):
        return [
            ("Loading project session...", lambda: self.session.worker if self.session is not None else None),
            ("Loading labeler module...", lambda: importlib.import_module("aligner_gui.labeler.labeler_widget")),
            ("Loading trainer module...", lambda: importlib.import_module("aligner_gui.trainer.trainer_widget")),
            ("Loading charts...", lambda: importlib.import_module("pyqtgraph")),
            ("Loading training backend...", StartupWarmupThread._import_training_stack),
        ]

    def _get_post_show_tester_steps(self):
        return [
            ("Loading tester module...", lambda: importlib.import_module("aligner_gui.tester.tester_widget")),
            ("Loading inference backend...", StartupWarmupThread._import_inference_stack),
        ]

    def _set_loading_message(self, message: str, splash=None):
        self.statusBar().showMessage(message)
        if splash is not None:
            splash.showMessage(
                message,
                QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
                QtCore.Qt.white,
            )
        QApplication.processEvents()

    def _run_import_warmup(self, steps, splash=None):
        thread = StartupWarmupThread(steps)
        thread.sig_progress.connect(lambda _cur, _total, message: self._set_loading_message(message, splash))
        thread.start()
        while thread.isRunning():
            QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            thread.wait(50)
        QApplication.processEvents()
        if thread.error_message:
            logging.warning("Startup warmup completed with warning: %s", thread.error_message)

    def _start_background_tester_warmup(self):
        if const.STARTUP_PREWARM_MODE != "startup":
            return
        if not self.app_vm.is_idle:
            return
        if self.TAB_TESTER in self._tab_widgets:
            return
        if len(self._background_warmup_threads) > 0:
            return

        thread = StartupWarmupThread(self._get_post_show_tester_steps())
        thread.sig_progress.connect(lambda _cur, _total, message: self.statusBar().showMessage(message, 1500))
        thread.finished.connect(lambda: self._on_background_warmup_finished(thread))
        self._background_warmup_threads.append(thread)
        thread.start()

    def _on_background_warmup_finished(self, thread: StartupWarmupThread):
        if thread in self._background_warmup_threads:
            self._background_warmup_threads.remove(thread)
        if thread.error_message:
            logging.warning("Background warmup completed with warning: %s", thread.error_message)
        elif self.get_app_status() == const.APP_STATUS_IDLE:
            self.statusBar().showMessage("Ready", 2000)

    def initialize_project_runtime(self, splash=None):
        if const.STARTUP_PREWARM_MODE == "startup":
            self._run_import_warmup(self._get_startup_warmup_steps(), splash)
            for idx, tab_name in enumerate((self.TAB_LABELER, self.TAB_TRAINER), start=1):
                self._set_loading_message(f"Preparing {tab_name} ({idx}/2)...", splash)
                self._get_tab_widget(tab_name)
            self._show_tab(self.TAB_LABELER, show_log=False)
            self.statusBar().showMessage("Ready", 3000)
            QtCore.QTimer.singleShot(3000, self._start_background_tester_warmup)
            return

        self.triggered_action_labeler()
        if const.STARTUP_PREWARM_MODE == "staged":
            QtCore.QTimer.singleShot(const.STARTUP_PREWARM_INITIAL_DELAY_MS, self._start_staged_prewarm)

    def _start_staged_prewarm(self):
        if const.STARTUP_PREWARM_MODE != "staged":
            return
        if self.session is None or not self.app_vm.is_idle:
            return
        self._prewarm_queue = ["session", self.TAB_TRAINER, self.TAB_TESTER]
        self._prewarm_active = True
        self._run_next_prewarm_step()

    def _run_next_prewarm_step(self):
        if not self._prewarm_active:
            return
        if self.session is None or not self.app_vm.is_idle:
            self._prewarm_active = False
            return
        if len(self._prewarm_queue) == 0:
            self._prewarm_active = False
            self.statusBar().showMessage("Ready", 3000)
            return

        step = self._prewarm_queue.pop(0)
        try:
            if step == "session":
                self.statusBar().showMessage("Preparing project session...")
                QApplication.processEvents()
                _ = self.session.worker
            elif step in (self.TAB_TRAINER, self.TAB_TESTER):
                if step not in self._tab_widgets:
                    self.statusBar().showMessage(f"Preparing {step}...")
                    QApplication.processEvents()
                    self._get_tab_widget(step)
        except Exception:
            logging.exception("Staged prewarm failed at step '%s'.", step)
            self._prewarm_active = False
            self.statusBar().showMessage("Ready", 3000)
            return

        QtCore.QTimer.singleShot(const.STARTUP_PREWARM_STEP_DELAY_MS, self._run_next_prewarm_step)

    def triggered_action_labeler(self):
        self._show_tab(self.TAB_LABELER, show_log=False)

    def triggered_action_trainer(self):
        self._show_tab(self.TAB_TRAINER, show_log=True)

    def triggered_action_test(self):
        self._show_tab(self.TAB_TESTER, show_log=True)

    def triggered_action_export(self):
        options = list_export_checkpoint_options(self._project_path)
        if not options:
            gui_util.get_message_box(self, "Invalid Export", "There is no trained model.")
            return

        selected_option = options[0]
        if len(options) == 2:
            msg = QMessageBox(self)
            msg.setWindowTitle("Export")
            msg.setText(
                "Which checkpoint do you want to export?\n"
                "The selected weights will be exported as model.pth for runtime compatibility."
            )
            btn_best = msg.addButton("Best checkpoint", QMessageBox.AcceptRole)
            btn_last = msg.addButton("Last checkpoint", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton(QMessageBox.Cancel)
            msg.exec_()

            clicked = msg.clickedButton()
            if clicked == btn_cancel:
                return
            if clicked == btn_last:
                selected_option = options[1]
            elif clicked == btn_best:
                selected_option = options[0]

        path_to_export = gui_util.get_open_dir_from_dialog(
            self,
            "Choose a directory to export model",
        )
        if path_to_export == "":
            return

        try:
            existing_files = os.listdir(path_to_export)
        except OSError:
            existing_files = []

        if existing_files:
            answer = gui_util.get_yes_no_box(
                self,
                "Export",
                "The export directory is not empty. Existing files with the same name may be overwritten.\nProceed?",
            )
            if answer == QMessageBox.No:
                return

        def work(processing_signal):
            return export_project_bundle(
                self._project_path,
                path_to_export,
                selected_option,
                processing_signal,
                project_config=self.session.get_project_config() if self.session is not None else None,
            )

        dlg = ProgressGeneralDialog("Exporting...", work, 6)
        dlg.exec_()
        if dlg.is_success:
            gui_util.get_message_box(
                self,
                "Export",
                f"Export succeeded.\nSelected checkpoint: {selected_option.source_name}\nSaved runtime weights as model.pth.",
            )
        elif not dlg.is_manual_stop:
            gui_util.get_message_box(self, "Export", "Export failed. Please check the log message.")

    def triggered_action_new_project(self):
        if not self.app_vm.is_idle:
            gui_util.get_message_box(self, "Busy", "Stop the current task before switching projects.")
            return

        import aligner_engine.const as aligner_engine_const
        import aligner_engine.utils as util

        new_project = ProjectOpenDialog()
        dir_path = new_project.get_dir_path("Create Project")
        if not new_project.create_project(dir_path):
            return

        if os.path.exists(util.join_path(new_project.project_path, aligner_engine_const.DIRNAME_AUTOSAVED)):
            yes_or_no = gui_util.get_yes_no_box(
                self,
                "Warning",
                "Project already exists in specified directory.\nAre you sure you want to proceed?",
            )
            if yes_or_no == QMessageBox.No:
                return
        self._replace_project(new_project.project_path, new_project.project_open_type)
        self.initialize_project_runtime()

    def triggered_action_load_project(self):
        if not self.app_vm.is_idle:
            gui_util.get_message_box(self, "Busy", "Stop the current task before switching projects.")
            return

        new_project = ProjectOpenDialog()
        path = new_project.get_dir_path("Load Project")
        if not new_project.load_project(path):
            return

        self._replace_project(new_project.project_path, new_project.project_open_type)
        self.initialize_project_runtime()

    def get_app_status(self) -> str:
        return self.app_vm.app_status

    def _on_app_status_changed(self, status: str) -> None:
        is_idle = status == const.APP_STATUS_IDLE
        self.action_export.setEnabled(is_idle)
        self.action_new_project.setEnabled(is_idle)
        self.action_load_project.setEnabled(is_idle)

    def closeEvent(self, event):
        yes_or_no = gui_util.get_yes_no_box(self, "Closing Question", "Are you sure you want to quit?")
        if yes_or_no == QMessageBox.Yes:
            self._stop_background_warmups()
            self._clear_tab_widgets()
            if self.session is not None:
                self.session.close()
            event.accept()
        else:
            event.ignore()
