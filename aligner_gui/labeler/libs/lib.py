from math import sqrt


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *



def newIcon(icon):
    return QIcon(':/' + icon)


def newButton(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)

class KeptBoxAction(QWidgetAction):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kept_box_info_class_name = ""
        self.kept_box_info_width = 0.0
        self.kept_box_info_height = 0.0

        self.kept_box_info_font_size = 8
        self.kept_box_info_name_height = 18
        self.kept_box_info_name_width = 42
        self.kept_box_info_value_height = 19
        self.kept_box_info_value_width = 60

        self.root_widget = QWidget()
        self.root_widget.setFixedHeight(72)
        self.root_widget.setFixedWidth(self.kept_box_info_name_width + self.kept_box_info_value_width + 30)
        self.root_widget.setStyleSheet("background-color: dark navy;")
        self.kept_box_info_layout = QHBoxLayout()
        self.title_and_kept_box_info_layout = QVBoxLayout()

        self.label_name_long = QLabel("Long : ", parent)
        self.label_name_long.setFont(QFont('Arial', self.kept_box_info_font_size))
        self.label_name_long.setFixedHeight(self.kept_box_info_name_height)
        self.label_name_long.setContentsMargins(0, 0, 0, 0)
        self.label_name_long.setFixedWidth(self.kept_box_info_name_width)

        self.label_name_short = QLabel("Short : ", parent)
        self.label_name_short.setFont(QFont('Arial', self.kept_box_info_font_size))
        self.label_name_short.setFixedHeight(self.kept_box_info_name_height)
        self.label_name_short.setContentsMargins(0, 0, 0, 0)
        self.label_name_short.setFixedWidth(self.kept_box_info_name_width)

        self.bounding_label_name_layout = QVBoxLayout()
        self.bounding_label_name_layout.addWidget(self.label_name_long)
        self.bounding_label_name_layout.addWidget(self.label_name_short)
        self.bounding_label_name_layout.setContentsMargins(0, 6, 0, 6)

        self.bounding_label_name = QWidget()
        self.bounding_label_name.setLayout(self.bounding_label_name_layout)
        self.bounding_label_name.setFixedWidth(self.kept_box_info_name_width)

        self.spinbox_value_long = QDoubleSpinBox(parent)
        self.spinbox_value_long.editingFinished.connect(self._long_value_changed_event)
        self.spinbox_value_long.setMaximum(float('inf'))
        self.spinbox_value_long.setMinimum(0.0)
        self.spinbox_value_long.setValue(0.0)
        self.spinbox_value_long.setDecimals(1)
        self.spinbox_value_long.setFont(QFont('Arial', self.kept_box_info_font_size))
        self.spinbox_value_long.setFixedHeight(self.kept_box_info_value_height)
        self.spinbox_value_long.setFixedWidth(self.kept_box_info_value_width)
        self.spinbox_value_long.setStyleSheet("QDoubleSpinBox { color: black;}")

        self.spinbox_value_short = QDoubleSpinBox(parent)
        self.spinbox_value_short.editingFinished.connect(self._short_value_changed_event)
        self.spinbox_value_short.setMaximum(float('inf'))
        self.spinbox_value_short.setMinimum(0.0)
        self.spinbox_value_short.setValue(0.0)
        self.spinbox_value_short.setDecimals(1)
        self.spinbox_value_short.setFont(QFont('Arial', self.kept_box_info_font_size))
        self.spinbox_value_short.setFixedHeight(self.kept_box_info_value_height)
        self.spinbox_value_short.setFixedWidth(self.kept_box_info_value_width)
        self.spinbox_value_short.setStyleSheet("QDoubleSpinBox { color: black;}")

        self.bounding_label_value_layout = QVBoxLayout()
        self.bounding_label_value_layout.addWidget(self.spinbox_value_long)
        self.bounding_label_value_layout.addWidget(self.spinbox_value_short)
        self.bounding_label_value_layout.setContentsMargins(0, 2, 0, 2)
        self.bounding_label_value = QWidget()
        self.bounding_label_value.setLayout(self.bounding_label_value_layout)
        self.bounding_label_value.setFixedWidth(self.kept_box_info_value_width + 5)

        self.bounding_label_name_layout.setContentsMargins(0, 0, 0, 0)
        self.bounding_label_value_layout.setContentsMargins(0, 0, 0, 0)

        self.kept_box_info_layout.addWidget(self.bounding_label_name)
        self.kept_box_info_layout.addWidget(self.bounding_label_value)
        self.kept_box_info_layout.setContentsMargins(5, 0, 5, 0)

        self.kept_box_info = QWidget()
        self.kept_box_info.setLayout(self.kept_box_info_layout)

        self.label_title = QLabel("Kept Box", parent)
        self.label_title.setFont(QFont('Arial', self.kept_box_info_font_size))
        self.label_title.setStyleSheet("QLabel { color: gray }")
        self.label_title.setFixedHeight(self.kept_box_info_name_height)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setContentsMargins(0, 0, 0, 0)

        self.title_and_kept_box_info_layout.addWidget(self.kept_box_info)
        self.title_and_kept_box_info_layout.addWidget(self.label_title)
        self.title_and_kept_box_info_layout.setContentsMargins(0, 0, 0, 0)
        self.root_widget.setLayout(self.title_and_kept_box_info_layout)

        self.spinbox_value_long.setStyleSheet("QDoubleSpinBox { background-color: rgb(25,35,45) }")
        self.spinbox_value_short.setStyleSheet("QDoubleSpinBox { background-color: rgb(25,35,45) }")
        self.setDefaultWidget(self.root_widget)

    def set_kept_box_info(self, long, short):
        self.spinbox_value_short.setValue(short)
        self.spinbox_value_long.setValue(long)

    def get_kept_box_info(self):
        return self.spinbox_value_long.value(), self.spinbox_value_short.value()

    def _short_value_changed_event(self):
        long = self.spinbox_value_long.value()
        short = self.spinbox_value_short.value()

        if long >= short:
            self.spinbox_value_long.setValue(long)
            self.spinbox_value_short.setValue(short)
        else:
            self.spinbox_value_long.setValue(short)
            self.spinbox_value_short.setValue(short)
        self.spinbox_with_value_zero_is_highlighted()

    def _long_value_changed_event(self):
        long = self.spinbox_value_long.value()
        short = self.spinbox_value_short.value()

        if long >= short:
            self.spinbox_value_long.setValue(long)
            self.spinbox_value_short.setValue(short)
        else:
            self.spinbox_value_long.setValue(long)
            self.spinbox_value_short.setValue(long)
        self.spinbox_with_value_zero_is_highlighted()

    def spinbox_with_value_zero_is_highlighted(self):
        if self.spinbox_value_long.value() == 0:
            self.spinbox_value_long.setStyleSheet("QDoubleSpinBox { background-color: rgb(255,200,0) }")
        else:
            self.spinbox_value_long.setStyleSheet("QDoubleSpinBox { background-color: rgb(25,35,45) }")
        if self.spinbox_value_short.value() == 0:
            self.spinbox_value_short.setStyleSheet("QDoubleSpinBox { background-color: rgb(255,200,0) }")
        else:
            self.spinbox_value_short.setStyleSheet("QDoubleSpinBox { background-color: rgb(25,35,45) }")



