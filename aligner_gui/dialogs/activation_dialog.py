from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from aligner_gui.ui.activation_dlg import Ui_activation_dlg

from aligner_engine import release_util
from aligner_engine.version import VERSION


class ActivationDialog(QDialog, Ui_activation_dlg):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowIcon(QtGui.QIcon("aligner_gui\\icons\\012-left-align-4.png"))

        self.btnOK.clicked.connect(self.clicked_btnOK)
        self.btnCancel.clicked.connect(self.clicked_btnCancel)
        self.editKey.textChanged.connect(self.changed_editKey)
        self.logo.setPixmap(QtGui.QPixmap("aligner_gui\\icons\\dice_aligner_title.png").scaled(470, 109))

        self._mac_addr = release_util.get_mac_addrs()[0]
        self._disk_id = release_util.get_disk_id()
        self.editInfo.setText('PC ID: ' + str(self._disk_id) + '\nDICE Aligner Version: ' + VERSION)
        self.btnOK.setEnabled(False)

    def clicked_btnOK(self):
        activ_path = release_util.get_activation_path()
        with open(activ_path, 'w') as f:
            f.write(self.editKey.text())
        self.accept()

    def clicked_btnCancel(self):
        self.close()

    def changed_editKey(self):
        key = self.editKey.text()
        result = release_util.SHACipher.activation_check_with_key(self._disk_id, key)
        self.btnOK.setEnabled(result)



