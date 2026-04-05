import os.path
import logging

from aligner_gui.ui.project_open_dlg import Ui_project_open_dlg
from PyQt5.QtWidgets import QDialog, QPushButton
from PyQt5 import QtGui, QtWidgets, QtCore
from aligner_gui.shared import const
from aligner_gui.shared import io_util
from aligner_gui.shared import gui_util
from aligner_engine.const import PROJECT_CONFIG_NAME
from aligner_engine.version import VERSION


class ProjectOpenDialog(QDialog, Ui_project_open_dlg):
    PATH_RECENT_LIST = io_util.join_path(io_util.get_aligner_home_dir(), 'dice_aligner_recent_project_list.txt')

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("aligner_gui\\icons\\012-left-align-4.png"))
        self.setWindowTitle(f"{const.APP_NAME} - v{VERSION}")

        self.btn_create_project.setEnabled(True)
        self.btn_create_project.clicked.connect(self._on_clicked_create_project)
        self.btn_create_project.setIcon(QtGui.QIcon("aligner_gui\\icons\\open_project.png"))
        self.btn_create_project.setIconSize(QtCore.QSize(27, 27))
        self.btn_create_project.setStyleSheet(
            "QPushButton{border-radius: 0;border:none;text-align:left;padding-left:8px;}"
            "QPushButton::hover{background-color:gray;font-weight: bold;}")

        self.btn_load_project.setEnabled(True)
        self.btn_load_project.clicked.connect(self._on_clicked_load_project)
        self.btn_load_project.setIcon(QtGui.QIcon("aligner_gui\\icons\\load_project.png"))
        self.btn_load_project.setIconSize(QtCore.QSize(27, 27))
        self.btn_load_project.setStyleSheet(
            "QPushButton{border-radius: 0;border:none;text-align:left;padding-left:8px;}"
            "QPushButton::hover{background-color:gray;font-weight: bold;}")

        self.project_path = ""
        self.project_open_type = ""

        self.recent_path_list = []
        self.refresh_recent_list(self.PATH_RECENT_LIST)

    def _on_clicked_create_project(self):
        dir_path = self.get_dir_path("Create Project")
        self.create_project(dir_path)

    def _on_clicked_load_project(self):
        dir_path = self.get_dir_path("Load Project")
        self.load_project(dir_path)

    def get_dir_path(self, caption: str):
        dir_path = gui_util.get_open_dir_from_dialog(self, caption)
        if dir_path == "":
            return None
        else:
            return os.path.abspath(dir_path)

    def create_project(self, path):
        if path is not None:
            if self.is_new_project(path):
                self.project_path = path
                self.project_open_type = const.PROJECT_OPEN_TYPE_NEW
                self._start_project()
                self.refresh_list()
                return True
        return False

    def load_project(self, path):
        if path is None:
            return False

        if self.is_new_project(path):
            gui_util.get_message_box(self, "Invalid Project", "Invalid Project")
            return False

        self.project_path = path
        self.project_open_type = const.PROJECT_OPEN_TYPE_LOAD
        self._start_project()
        self.refresh_list()
        return True

    def _start_project(self):
        self.close()

    def add_btn_load_project(self, path):
        path = os.path.abspath(path)
        if not io_util.is_exist(path) or path in self.recent_path_list:
            return False

        base_config_path = io_util.join_path(path, PROJECT_CONFIG_NAME)
        if not io_util.is_exist(base_config_path):
            return False

        self.recent_path_list.append(path)

        proj_name = os.path.basename(os.path.normpath(path))
        import time
        mtime = time.localtime(os.path.getmtime(base_config_path))
        mtime = time.strftime('%Y-%m-%d %H:%M', mtime)

        btn_load_project = QPushButton()
        style = ("QPushButton{border-radius: 0;border:none;text-align:left;background-color:dark navy;}"
                 "QPushButton::hover{background-color:gray;font-weight: bold;}")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_load_project.setStyleSheet(style)
        btn_load_project.setSizePolicy(sizePolicy)
        btn_load_project.sizeIncrement()
        btn_load_project.setText(f'{proj_name}\n{path}\nlatest update: {mtime}')
        btn_load_project.setIcon(QtGui.QIcon("aligner_gui\\icons\\prj.png"))
        btn_load_project.clicked.connect(lambda: self.load_project(path))
        self.layoutRecentList.addWidget(btn_load_project)

        return True

    def refresh_recent_list(self, path: str):
        try:
            if not io_util.is_exist(path):
                os.makedirs(io_util.get_aligner_home_dir(), exist_ok=True)
                with open(path, 'w'):
                    pass

            files = io_util.read_lines(path)
            available_files = []
            for file_path in files:
                if self.add_btn_load_project(file_path.strip()):
                    available_files.append(file_path)

            io_util.write_lines(path, available_files)

            spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.layoutRecentList.addSpacerItem(spacerItem)

        except PermissionError:
            logging.warning("Permission denied. Unable to create file: %s", path)
        except FileNotFoundError:
            logging.warning("Failed to create file: %s. Directory path is incorrect.", path)
        except Exception as e:
            logging.exception("Unexpected error while refreshing recent project list: %s", e)

    def refresh_list(self):
        try:
            lists = io_util.read_lines(self.PATH_RECENT_LIST)
            if self.project_path in lists:
                lists.remove(self.project_path)

            lists.insert(0, self.project_path)
            io_util.write_lines(self.PATH_RECENT_LIST, lists)

        except PermissionError:
            logging.warning("Permission denied. Unable to create file: %s", self.PATH_RECENT_LIST)
        except Exception as e:
            logging.exception("Unexpected error while refreshing recent project cache: %s", e)

    def is_new_project(self, path):
        if not io_util.is_exist(path):
            return True

        base_config_path = io_util.join_path(path, PROJECT_CONFIG_NAME)
        if not io_util.is_exist(base_config_path):
            return True

        if path not in self.recent_path_list:
            self.recent_path_list.append(path)

        return False

