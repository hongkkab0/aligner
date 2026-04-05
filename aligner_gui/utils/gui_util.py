import cv2
import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import csv
from aligner_gui.utils import const
import numpy as np

# from utils import utils

dark_style = None
SUPPORTED_IMAGE_FORMATS = ['.bmp', '.cur', '.gif', '.icns', '.ico', '.jpeg', '.jpg', '.pbm', '.pgm', '.png', '.ppm',
                           '.svg', '.svgz', '.tga', '.tif', '.tiff', '.wbmp', '.webp', '.xbm', '.xpm']
SUPPORTED_IMAGE_FORMATS_WITHOUT_DOT = ['bmp', 'cur', 'gif', 'icns', 'ico', 'jpeg', 'jpg', 'pbm', 'pgm', 'png', 'ppm',
                           'svg', 'svgz', 'tga', 'tif', 'tiff', 'wbmp', 'webp', 'xbm', 'xpm']


def get_dark_style():
    global dark_style
    if dark_style is None:
        dark_style = qdarkstyle.load_stylesheet_pyqt5()
    return dark_style


def show_message_box(title: str, message: str):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec_()


def reset_layout(layout: QLayout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().close()
        layout.takeAt(i)


def hide_layout(layout: QLayout):
    for i in reversed(range(layout.count())):
        widget = layout.itemAt(i).widget()
        widget.hide()
        layout.removeWidget(widget)


def update_layout(layout: QLayout, widget: QWidget):
    hide_layout(layout)
    if widget is not None:
        layout.addWidget(widget)
        widget.show()


def get_filelist_from_dialog(parent, caption="Open Files"):
    selectedList, _ = QFileDialog.getOpenFileNames(parent=parent, caption=caption)
    return selectedList


def get_file_from_dialog(parent, is_open, ext_list, save_file_name = ""):
    filter_dlg = "target file ("
    for ext in ext_list:
        filter_dlg += "*." + ext + " "
    filter_dlg = filter_dlg.rstrip()
    filter_dlg += ")"

    if is_open:
        file_name, _ = QFileDialog.getOpenFileName(parent, "Open File", "", filter_dlg)
    else:
        file_name, _ = QFileDialog.getSaveFileName(parent, "Save File", save_file_name, filter_dlg)
    return file_name

def get_files_from_dialog(parent, is_open, ext_list):
    filter_dlg = "target file ("
    for ext in ext_list:
        filter_dlg += "*." + ext + " "
    filter_dlg = filter_dlg.rstrip()
    filter_dlg += ")"

    if is_open:
        file_names = QFileDialog.getOpenFileNames(parent, "Open File", "", filter_dlg)
    else:
        file_names = QFileDialog.getSaveFileNames(parent, "Save File", "", filter_dlg)
    return file_names[0]


def get_path_from_dialog(parent, caption="Open Folder"):
    file_name = QFileDialog.getExistingDirectory(parent, caption, "")
    return file_name if len(file_name) > 0 else ''


def get_open_path_from_dialog(parent, caption="Open Folder"):
    return get_path_from_dialog(parent, caption)

def get_open_dir_from_dialog(parent_widget, caption):
    default_open_dir_path = "."
    target_dir_path = str(
        QFileDialog.getExistingDirectory(
            parent_widget,
            parent_widget.tr("{0} - {1}".format(const.APP_NAME, caption)),
            default_open_dir_path,
            QFileDialog.ShowDirsOnly
            | QFileDialog.DontResolveSymlinks,
        )
    )
    return target_dir_path


# nullable
def get_first_widget(layout: QLayout):
    item = layout.itemAt(0)
    if item is None:
        return None

    return item.widget()


def get_message_box(widget, title='Message Box', msg='Message'):
    msgBox = QMessageBox.information(
        widget, title, msg,
        QMessageBox.Ok,
        QMessageBox.Ok
    )
    return msgBox


def get_yes_no_box(widget, title='YesNo Box', msg='Message'):
    msgBox = QMessageBox.question(
        widget, title, msg,
    )
    return msgBox


def get_custom_box(widget, yes, no, use_cancel=True, title='Custom Box', msg='Message'):
    msgBox = QMessageBox(widget)
    msgBox.addButton(QPushButton(yes), QMessageBox.YesRole)
    msgBox.addButton(QPushButton(no), QMessageBox.NoRole)
    if use_cancel:
        msgBox.addButton(QMessageBox.Cancel)
    msgBox.setWindowTitle(title)
    msgBox.setText(msg)
    return msgBox.exec()


def table2csv(table: QTableWidget, file_name):
    try:
        with open(file_name, 'w', newline='') as f:
            w = csv.writer(f)

            nCols = table.columnCount()
            nRows = table.rowCount()
            line_text = ['']
            # header
            for c in range(nCols):
                header = table.horizontalHeaderItem(c)
                if header is not None:
                    text = str(header.text()).replace('\n', ' ')
                    line_text.append(text)
                else:
                    line_text.append('')
            w.writerow(line_text)

            # contents
            for r in range(0, nRows):
                header = table.verticalHeaderItem(r)
                if header is not None:
                    text = str(header.text()).replace('\n', ' ')
                    line_text = [text]
                else:
                    line_text = [str(r)]

                for c in range(0, nCols):
                    item = table.item(r, c)
                    if item is not None:
                        text = str(item.text()).replace('\n', ' ')
                        line_text.append(text)
                    else:
                        line_text.append('')
                w.writerow(line_text)
    except Exception as e:
        print(e)

def cvimg_to_pixmap(cvimg):
    try:
        height, width, channel = cvimg.shape
        bytesPerLine = channel * width
        qImg = QImage(cvimg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        return pixmap
    except Exception as e:
        print(e)
        return None

