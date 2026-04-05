from __future__ import annotations

import importlib
import logging
import os
from typing import Dict, TYPE_CHECKING

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox

from aligner_gui import __appname__
from aligner_gui.dialogs.project_open_dialog import ProjectOpenDialog
from aligner_gui.project.project_export_service import export_project_bundle, list_export_checkpoint_options
from aligner_gui.project.project_session import ProjectSession
from aligner_gui.shared import const, gui_util
from aligner_gui.shared.log_manager import configure_project_logging
from aligner_gui.viewmodels.base_viewmodel import ViewModelBase
from aligner_gui.shared.progress_general_dialog import ProgressGeneralDialog

if TYPE_CHECKING:
    from aligner_gui.main_window import MainWindow


class StartupWarmupThread(QtCore.QThread):
    sig_progress = QtCore.pyqtSignal(int, int, str)

    def __init__(self, steps):
        super().__init__()
        self._steps = list(steps)
        self.error_message = ""

    @staticmethod
    def import_training_stack():
        import mmcv  # noqa: F401
        import torch  # noqa: F401
        from mmengine.config import Config  # noqa: F401
        from mmdet.registry import RUNNERS  # noqa: F401
        from mmrotate.registry import HOOKS  # noqa: F401

    @staticmethod
    def import_inference_stack():
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


class MainWindowViewModel(ViewModelBase):
    TAB_LABELER = "labeler"
    TAB_TRAINER = "trainer"
    TAB_TESTER = "tester"

    # Emitted whenever the app-wide busy state changes (True = training/busy).
    # MainWindow connects this to enable/disable toolbar actions.
    app_status_changed = QtCore.pyqtSignal(bool)

    def __init__(self, view: 'MainWindow', project_path: str, project_open_type: str):
        super().__init__(view)
        self.view = view
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
        self.status_message_requested.connect(self.view.statusBar().showMessage)
        self._init_view()
        self._replace_project(project_path, project_open_type)
        self.set_app_status_idle()

    @property
    def project_path(self) -> str:
        return self._project_path

    def _init_view(self):
        # Connect action signals to ViewModel command methods.
        # Icon / widget setup is handled in MainWindow._init_ui() (a View concern).
        self.view.action_labeler.triggered.connect(self.show_labeler)
        self.view.action_trainer.triggered.connect(self.show_trainer)
        self.view.action_test.triggered.connect(self.show_tester)
        self.view.action_export.triggered.connect(self.export_project)
        self.view.action_new_project.triggered.connect(self.create_new_project)
        self.view.action_load_project.triggered.connect(self.load_existing_project)

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
        self.view.setWindowTitle(__appname__ + " - " + project_path)
        self.view.log_widget_container.hide()
        gui_util.update_layout(self.view.layout_main, self._loading_label)

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
        gui_util.hide_layout(self.view.layout_main)

    def _create_tab_widget(self, tab_name: str):
        assert self.session is not None

        if tab_name == self.TAB_LABELER:
            from aligner_gui.labeler.labeler_view import LabelerView
            return LabelerView(self.view, self.session, self._project_path, is_new=self._is_new)
        if tab_name == self.TAB_TESTER:
            from aligner_gui.tester.tester_view import TesterView
            return TesterView(self.view, self.session, is_new=self._is_new)
        if tab_name == self.TAB_TRAINER:
            from aligner_gui.trainer.trainer_view import TrainerView
            return TrainerView(self.view, self.session, tester_reload_callback=self._reload_tester_if_loaded)
        raise NotImplementedError(tab_name)

    def _get_tab_widget(self, tab_name: str):
        widget = self._tab_widgets.get(tab_name)
        if widget is not None:
            return widget

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.view.statusBar().showMessage(f"Loading {tab_name}...")
        QApplication.processEvents()
        try:
            widget = self._create_tab_widget(tab_name)
            self._tab_widgets[tab_name] = widget
            return widget
        finally:
            QApplication.restoreOverrideCursor()
            self.view.statusBar().showMessage("Ready", 3000)

    def show_tab(self, tab_name: str, show_log: bool):
        if show_log:
            self.view.log_widget_container.show()
            if tab_name == self.TAB_TRAINER:
                gui_util.update_layout(self.view.layout_log, self.view.trainer_log_widget)
            elif tab_name == self.TAB_TESTER:
                gui_util.update_layout(self.view.layout_log, self.view.tester_log_widget)
        else:
            self.view.log_widget_container.hide()

        widget = self._get_tab_widget(tab_name)
        gui_util.update_layout(self.view.layout_main, widget)
        self.view.action_labeler.setChecked(tab_name == self.TAB_LABELER)
        self.view.action_trainer.setChecked(tab_name == self.TAB_TRAINER)
        self.view.action_test.setChecked(tab_name == self.TAB_TESTER)
        self.view.setWindowTitle(__appname__ + " - " + self._project_path)

    def _reload_tester_if_loaded(self):
        tester_view = self._tab_widgets.get(self.TAB_TESTER)
        if tester_view is not None:
            tester_view.reload_file_list()

    def _get_startup_warmup_steps(self):
        return [
            ("Loading project session...", lambda: self.session.worker if self.session is not None else None),
            ("Loading labeler module...", lambda: importlib.import_module("aligner_gui.labeler.labeler_view")),
            ("Loading trainer module...", lambda: importlib.import_module("aligner_gui.trainer.trainer_view")),
            ("Loading charts...", lambda: importlib.import_module("pyqtgraph")),
            ("Loading training backend...", StartupWarmupThread.import_training_stack),
        ]

    def _get_post_show_tester_steps(self):
        return [
            ("Loading tester module...", lambda: importlib.import_module("aligner_gui.tester.tester_view")),
            ("Loading inference backend...", StartupWarmupThread.import_inference_stack),
        ]

    def _set_loading_message(self, message: str, splash=None):
        self.view.statusBar().showMessage(message)
        if splash is not None:
            splash.showMessage(message, QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, QtCore.Qt.white)
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
        if self.get_app_status() != const.APP_STATUS_IDLE:
            return
        if self.TAB_TESTER in self._tab_widgets:
            return
        if len(self._background_warmup_threads) > 0:
            return

        thread = StartupWarmupThread(self._get_post_show_tester_steps())
        thread.sig_progress.connect(lambda _cur, _total, message: self.view.statusBar().showMessage(message, 1500))
        thread.finished.connect(lambda: self._on_background_warmup_finished(thread))
        self._background_warmup_threads.append(thread)
        thread.start()

    def _on_background_warmup_finished(self, thread: StartupWarmupThread):
        if thread in self._background_warmup_threads:
            self._background_warmup_threads.remove(thread)
        if thread.error_message:
            logging.warning("Background warmup completed with warning: %s", thread.error_message)
        elif self.get_app_status() == const.APP_STATUS_IDLE:
            self.view.statusBar().showMessage("Ready", 2000)

    def initialize_project_runtime(self, splash=None):
        if const.STARTUP_PREWARM_MODE == "startup":
            self._run_import_warmup(self._get_startup_warmup_steps(), splash)
            for idx, tab_name in enumerate((self.TAB_LABELER, self.TAB_TRAINER), start=1):
                self._set_loading_message(f"Preparing {tab_name} ({idx}/2)...", splash)
                self._get_tab_widget(tab_name)
            self.show_tab(self.TAB_LABELER, show_log=False)
            self.view.statusBar().showMessage("Ready", 3000)
            QtCore.QTimer.singleShot(3000, self._start_background_tester_warmup)
            return

        self.show_labeler()
        if const.STARTUP_PREWARM_MODE == "staged":
            QtCore.QTimer.singleShot(const.STARTUP_PREWARM_INITIAL_DELAY_MS, self._start_staged_prewarm)

    def _start_staged_prewarm(self):
        if const.STARTUP_PREWARM_MODE != "staged":
            return
        if self.session is None or self.get_app_status() != const.APP_STATUS_IDLE:
            return
        self._prewarm_queue = ["session", self.TAB_TRAINER, self.TAB_TESTER]
        self._prewarm_active = True
        self._run_next_prewarm_step()

    def _run_next_prewarm_step(self):
        if not self._prewarm_active:
            return
        if self.session is None or self.get_app_status() != const.APP_STATUS_IDLE:
            self._prewarm_active = False
            return
        if len(self._prewarm_queue) == 0:
            self._prewarm_active = False
            self.view.statusBar().showMessage("Ready", 3000)
            return

        step = self._prewarm_queue.pop(0)
        try:
            if step == "session":
                self.view.statusBar().showMessage("Preparing project session...")
                QApplication.processEvents()
                _ = self.session.worker
            elif step in (self.TAB_TRAINER, self.TAB_TESTER):
                if step not in self._tab_widgets:
                    self.view.statusBar().showMessage(f"Preparing {step}...")
                    QApplication.processEvents()
                    self._get_tab_widget(step)
        except Exception:
            logging.exception("Staged prewarm failed at step '%s'.", step)
            self._prewarm_active = False
            self.view.statusBar().showMessage("Ready", 3000)
            return

        QtCore.QTimer.singleShot(const.STARTUP_PREWARM_STEP_DELAY_MS, self._run_next_prewarm_step)

    def show_labeler(self):
        self.show_tab(self.TAB_LABELER, show_log=False)

    def show_trainer(self):
        self.show_tab(self.TAB_TRAINER, show_log=True)

    def show_tester(self):
        self.show_tab(self.TAB_TESTER, show_log=True)

    def export_project(self):
        options = list_export_checkpoint_options(self._project_path)
        if not options:
            gui_util.get_message_box(self.view, "Invalid Export", "There is no trained model.")
            return

        selected_option = options[0]
        if len(options) == 2:
            msg = QMessageBox(self.view)
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

        path_to_export = gui_util.get_open_dir_from_dialog(self.view, "Choose a directory to export model")
        if path_to_export == "":
            return

        try:
            existing_files = os.listdir(path_to_export)
        except OSError:
            existing_files = []

        if existing_files:
            answer = gui_util.get_yes_no_box(
                self.view,
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
                self.view,
                "Export",
                f"Export succeeded.\nSelected checkpoint: {selected_option.source_name}\nSaved runtime weights as model.pth.",
            )
        elif not dlg.is_manual_stop:
            gui_util.get_message_box(self.view, "Export", "Export failed. Please check the log message.")

    def create_new_project(self):
        if self.get_app_status() != const.APP_STATUS_IDLE:
            gui_util.get_message_box(self.view, "Busy", "Stop the current task before switching projects.")
            return

        import aligner_engine.const as aligner_engine_const
        import aligner_engine.utils as util

        new_project = ProjectOpenDialog()
        dir_path = new_project.get_dir_path("Create Project")
        if not new_project.create_project(dir_path):
            return

        if os.path.exists(util.join_path(new_project.project_path, aligner_engine_const.DIRNAME_AUTOSAVED)):
            yes_or_no = gui_util.get_yes_no_box(
                self.view,
                "Warning",
                "Project already exists in specified directory.\nAre you sure you want to proceed?",
            )
            if yes_or_no == QMessageBox.No:
                return
        self._replace_project(new_project.project_path, new_project.project_open_type)
        self.initialize_project_runtime()

    def load_existing_project(self):
        if self.get_app_status() != const.APP_STATUS_IDLE:
            gui_util.get_message_box(self.view, "Busy", "Stop the current task before switching projects.")
            return

        new_project = ProjectOpenDialog()
        path = new_project.get_dir_path("Load Project")
        if not new_project.load_project(path):
            return

        self._replace_project(new_project.project_path, new_project.project_open_type)
        self.initialize_project_runtime()

    def get_app_status(self) -> str:
        return self._app_status

    def set_app_status_idle(self) -> None:
        self._app_status = const.APP_STATUS_IDLE
        self.app_status_changed.emit(False)

    def set_app_status_training(self) -> None:
        self._app_status = const.APP_STATUS_TRAINING
        self.app_status_changed.emit(True)

    def handle_close_event(self, event):
        yes_or_no = gui_util.get_yes_no_box(self.view, "Closing Question", "Are you sure you want to quit?")
        if yes_or_no == QMessageBox.Yes:
            self._stop_background_warmups()
            self._clear_tab_widgets()
            if self.session is not None:
                self.session.close()
            event.accept()
        else:
            event.ignore()

