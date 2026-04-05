import os.path as osp
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

def ImageFile2LabelFile(image_file: str):
    if image_file.lower().endswith(".json"):
        return image_file
    return osp.splitext(image_file)[0] + ".json"


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read_file(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default