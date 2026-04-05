from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow

from aligner_gui.viewmodels.main_viewmodel import MainWindowViewModel
from aligner_gui.ui.main_window import Ui_main_window


class MainWindow(QMainWindow, Ui_main_window):
    def __init__(self, project_path, project_open_type):
        super().__init__()
        self.setupUi(self)
        self.viewmodel = MainWindowViewModel(self, project_path, project_open_type)

    def initialize_project_runtime(self, splash=None):
        self.viewmodel.initialize_project_runtime(splash)

    def triggered_action_labeler(self):
        self.viewmodel.show_labeler()

    def triggered_action_trainer(self):
        self.viewmodel.show_trainer()

    def triggered_action_test(self):
        self.viewmodel.show_tester()

    def triggered_action_export(self):
        self.viewmodel.export_project()

    def triggered_action_new_project(self):
        self.viewmodel.create_new_project()

    def triggered_action_load_project(self):
        self.viewmodel.load_existing_project()

    def get_app_status(self) -> str:
        return self.viewmodel.get_app_status()

    def set_app_status_idle(self):
        self.viewmodel.set_app_status_idle()

    def set_app_status_training(self):
        self.viewmodel.set_app_status_training()

    def closeEvent(self, event):
        self.viewmodel.handle_close_event(event)
