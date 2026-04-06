import sys
import os
import ctypes
from PyQt5 import QtGui
from PyQt5.QtWidgets import *


APP_ICON_PATH = "aligner_gui\\icons\\012-left-align-4.png"
TITLE_IMAGE_PATH = "aligner_gui\\icons\\dice_aligner_title.png"
WINDOWS_APP_ID = "DICE.Aligner.App"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(PACKAGE_DIR)


def _set_windows_app_user_model_id():
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass


def show_busy_popup(app, message: str, title: str = "DICE Aligner"):
    progress = QProgressDialog()
    progress.setWindowTitle(title)
    progress.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
    progress.setLabelText(message)
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setRange(0, 0)
    progress.setWindowModality(False)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.show()
    app.processEvents()
    return progress


def ask_project_open():
    from aligner_gui.dialogs.project_open_dialog import ProjectOpenDialog

    project_open_dialog = ProjectOpenDialog()
    project_open_dialog.exec_()

    return project_open_dialog.project_path, project_open_dialog.project_open_type


def check_activation(app):
    from aligner_engine import release_util
    from aligner_gui.dialogs.activation_dialog import ActivationDialog

    popup = show_busy_popup(app, "Checking activation...")
    try:
        active_key = release_util.get_activation_key()
        if active_key is not None:
            if release_util.activation_check():
                print('Activation is confirmed')
                return True
    finally:
        popup.close()
        app.processEvents()

    dlg = ActivationDialog()
    if dlg.exec_():
        print('Activation is succeed')
        return True
    else:
        print('Activation is failed')
        return False


def make_app():
    from aligner_gui.shared import gui_util
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont

    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName("DICE Aligner")
    app.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
    # setup stylesheet
    font = QFont("Segoe UI", 9)
    # font.setStyleHint(QFont.Helvetica)
    app.setFont(font)
    app.setStyleSheet(gui_util.get_dark_style())

    return app


def show_main_window(app, project_path, project_open_type):

    from PyQt5 import QtCore

    pixmap = QtGui.QPixmap(TITLE_IMAGE_PATH)
    splash = QSplashScreen(pixmap)
    splash.setWindowIcon(QtGui.QIcon(APP_ICON_PATH))
    splash.setWindowState(splash.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    splash.activateWindow()
    splash.showMessage("Loading project...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, QtCore.Qt.white)
    splash.show()
    app.processEvents()
    from aligner_gui.main_window import MainWindow

    main_window = MainWindow(project_path, project_open_type)
    main_window.initialize_project_runtime(splash)
    main_window.show()
    main_window.raise_()
    splash.finish(main_window)
    app.exec_()


def main():
    app = make_app()

    if check_activation(app):
        project_path, project_open_type = ask_project_open()
        if project_path == "":
            return
        show_main_window(app, project_path, project_open_type)
    else:
        return

if __name__ == '__main__':
    if ROOT_PATH not in sys.path:
        sys.path.insert(0, ROOT_PATH)

    # This is for openvino
    venv_scripts = os.path.join(ROOT_PATH, '.venv', "Scripts")
    if os.path.isdir(venv_scripts):
        os.environ["PATH"] = os.environ["PATH"] + ";" + venv_scripts

    qt_plugin_path = os.path.join(
        ROOT_PATH,
        '.venv',
        'Lib',
        'site-packages',
        'PyQt5',
        'Qt',
        'plugins',
        'platforms',
    )
    if os.path.isdir(qt_plugin_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

    # sys.stdout = open(os.path.join(ROOT_PATH, "dice_aligner_out.log"), 'w')
    # sys.stderr = open(os.path.join(ROOT_PATH, "dice_aligner_err.log"), 'w')

    # startupinfo = subprocess.STARTUPINFO()
    # startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    # out, err = subprocess.Popen(
    #     cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo).communicate()

    os.chdir(ROOT_PATH)  # change working directory
    print("sys.path:", sys.path)
    main_window = main()

