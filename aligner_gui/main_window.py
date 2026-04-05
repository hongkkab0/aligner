from __future__ import annotations

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow

from aligner_gui.viewmodels.main_viewmodel import MainWindowViewModel
from aligner_gui.ui.main_window import Ui_main_window


class MainWindow(QMainWindow, Ui_main_window):
    """Top-level application window.

    Responsibilities
    ----------------
    * Own and set up all UI elements that are not part of a tab widget
      (icons, log panel, status bar).
    * Create the ViewModel and connect its signals to local UI-update slots.
    * Delegate close and action events to the ViewModel.

    The window holds **no** business logic. Every user action is forwarded to
    MainWindowViewModel; every state change comes back via a pyqtSignal.
    """

    def __init__(self, project_path: str, project_open_type: str) -> None:
        super().__init__()
        self.setupUi(self)
        self._init_ui()

        self.viewmodel = MainWindowViewModel(self, project_path, project_open_type)
        self.viewmodel.status_message_requested.connect(self.statusBar().showMessage)
        self.viewmodel.app_status_changed.connect(self._on_app_status_changed)
        self.viewmodel.tab_switched.connect(self._on_tab_switched)
        self.viewmodel.window_title_changed.connect(self.setWindowTitle)

    # ------------------------------------------------------------------
    # View initialisation (pure UI concern — no business logic here)
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        from aligner_gui.shared.log_widget import LogWidget
        from aligner_gui.shared import gui_util

        # Log panel widgets attached to the window (shared across tabs)
        self.trainer_log_widget = LogWidget(prefixes=("aligner.trainer",))
        self.tester_log_widget = LogWidget(prefixes=("aligner.tester",))
        gui_util.update_layout(self.layout_log, self.trainer_log_widget)
        self.log_widget_container.hide()

        # Icons
        self.setWindowIcon(QIcon("aligner_gui\\icons\\012-left-align-4.png"))
        self.action_labeler.setIcon(QIcon("aligner_gui\\icons\\essential\\bookmark(white).png"))
        self.action_trainer.setIcon(QIcon("aligner_gui\\icons\\essential\\002-book(white).png"))
        self.action_test.setIcon(QIcon("aligner_gui\\icons\\essential\\list(white).png"))
        self.action_export.setIcon(QIcon("aligner_gui\\icons\\essential\\078-folder-73(white).png"))

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # ViewModel signal slots (all UI updates live here, not in VM)
    # ------------------------------------------------------------------

    def _on_tab_switched(self, tab_name: str, widget, show_log: bool) -> None:
        from aligner_gui.shared import gui_util

        if show_log:
            self.log_widget_container.show()
            if tab_name == self.viewmodel.TAB_TRAINER:
                gui_util.update_layout(self.layout_log, self.trainer_log_widget)
            elif tab_name == self.viewmodel.TAB_TESTER:
                gui_util.update_layout(self.layout_log, self.tester_log_widget)
        else:
            self.log_widget_container.hide()

        gui_util.update_layout(self.layout_main, widget)
        self.action_labeler.setChecked(tab_name == self.viewmodel.TAB_LABELER)
        self.action_trainer.setChecked(tab_name == self.viewmodel.TAB_TRAINER)
        self.action_test.setChecked(tab_name == self.viewmodel.TAB_TESTER)

    def _on_app_status_changed(self, is_busy: bool) -> None:
        self.action_export.setEnabled(not is_busy)
        self.action_new_project.setEnabled(not is_busy)
        self.action_load_project.setEnabled(not is_busy)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_project_runtime(self, splash=None) -> None:
        self.viewmodel.initialize_project_runtime(splash)

    def get_app_status(self) -> str:
        return self.viewmodel.get_app_status()

    def set_app_status_idle(self) -> None:
        self.viewmodel.set_app_status_idle()

    def set_app_status_training(self) -> None:
        self.viewmodel.set_app_status_training()

    def closeEvent(self, event) -> None:
        self.viewmodel.handle_close_event(event)
