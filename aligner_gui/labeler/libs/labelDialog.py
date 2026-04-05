
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from aligner_gui.labeler.libs.lib import newIcon, labelValidator

BB = QDialogButtonBox


class LabelDialog(QDialog):

    def __init__(self, parent, label_names: dict):
        super(LabelDialog, self).__init__(parent)
        self.setWindowTitle("DICE Aligner")
        self._line_edit = QLineEdit()
        self._line_edit.setText('')
        self._line_edit.setValidator(labelValidator())
        self._line_edit.editingFinished.connect(self._post_process)
        layout = QVBoxLayout()
        layout.addWidget(self._line_edit)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self._validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self._list_widget = QListWidget(self)
        self._list_widget.itemClicked.connect(self._list_item_clicked)
        self._list_widget.itemDoubleClicked.connect(self._list_item_double_clicked)
        layout.addWidget(self._list_widget)

        for item in label_names.keys():
            self._list_widget.addItem(item)


        self.setLayout(layout)

    def _validate(self):
        try:
            if self._line_edit.text().trimmed():
                self.accept()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            if self._line_edit.text().strip():
                self.accept()

    def _post_process(self):
        try:
            self._line_edit.setText(self._line_edit.text().trimmed())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self._line_edit.setText(self._line_edit.text())

    def _list_item_clicked(self):
        text = self._list_widget.currentItem().text()
        self._line_edit.setText(text)

    def _list_item_double_clicked(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self._line_edit.setText(text)
        self._validate()

    # This is called when a shape is created or a shape is edited.
    def popUp(self, text='', move=True):
        self._line_edit.setText(text)
        self._line_edit.setSelection(0, len(text))
        self._line_edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self._line_edit.text() if self.exec_() else None
