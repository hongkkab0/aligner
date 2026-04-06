from PyQt5.QtGui import QImage

import os.path
import sys
import math
from aligner_gui.labeler.utils import ImageFile2LabelFile
from aligner_gui.labeler.libs.shape import Shape, SHAPE_TYPE_RO_RECTANGLE, SHAPE_TYPE_RECTANGLE
from typing import List, Set, Dict, Tuple
import json
from aligner_gui import __appname__
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from aligner_gui.shared import io_util
import cv2
import numpy as np



class LabelFileError(Exception):
    pass

# This object is created instantly when a file loaded and saved
class LabelFile(object):
    SUFFIX = ".json"
    LABEL_FORMAT_VERSION = "0.3.0"
    TASK_TYPE_ROTATE_DET = "rotate_det"

    def __init__(self, image_path):
        self._task_type = self.TASK_TYPE_ROTATE_DET
        self._shapes: List[Shape] = []
        self.image_path = image_path
        self._label_path = ImageFile2LabelFile(image_path)
        self._split = "none"
        self._is_need_confirm = False

    def existLabel(self):
        return self._label_path is not None and os.path.exists(self._label_path)

    def getLabelPath(self):
        return self._label_path

    def existImage(self):
        return os.path.exists(self.image_path)

    def load_label(self):
        try:
            with open(self._label_path, "r") as f:
                close = f
                data = json.load(f)

            labeler = data.get("labeler")
            label_format_version = data.get("label_format_version")
            task_type = data.get("task_type")
            image_height = data.get("imageHeight")
            image_width = data.get("imageWidth")
            image_depth = data.get("imageDepth")
            split = data.get("split")
            data_shapes = data.get("shapes")
            self._is_need_confirm = False if (data.get("need_confirm") == False or
                                              data.get("need_confirm") != True) else True
            shapes = []
            for data_shape in data_shapes:
                shape = Shape(label=data_shape['label'])
                shape.addPoint(QPointF(data_shape['x1'], data_shape['y1']))
                shape.addPoint(QPointF(data_shape['x2'], data_shape['y2']))
                shape.addPoint(QPointF(data_shape['x3'], data_shape['y3']))
                shape.addPoint(QPointF(data_shape['x4'], data_shape['y4']))
                center, (w, h), angle = cv2.minAreaRect(
                    np.array([[data_shape['x1'], data_shape['y1']],
                              [data_shape['x2'], data_shape['y2']],
                              [data_shape['x3'], data_shape['y3']],
                              [data_shape['x4'], data_shape['y4']]], dtype=np.float32))
                if angle == 90:
                    angle = 0
                angle = angle * math.pi / 180

                shape.direction = angle
                shape.isRotated = True
                shape.close()
                shapes.append(shape)
            self._shapes = shapes

        except Exception as e:
            self._shapes = []
            return

    def remove_label_file(self):
        io_util.remove_file(self._label_path)

    def save_label(self, image_info):
        def _format_shape(shape: Shape):
            return dict(label=shape.get_label(),
                        points=[(p.x(), p.y()) for p in shape.points],
                        center=shape.center,
                        isRotated=shape.isRotated)

        shapes = [_format_shape(shape) for shape in self._shapes]
        data_shapes = []
        for shape in shapes:
            x1 = shape["points"][0][0]
            y1 = shape["points"][0][1]
            x2 = shape["points"][1][0]
            y2 = shape["points"][1][1]
            x3 = shape["points"][2][0]
            y3 = shape["points"][2][1]
            x4 = shape["points"][3][0]
            y4 = shape["points"][3][1]

            data_shape = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4}
            data_shape["label"] = shape["label"]
            data_shape["shape_type"] = SHAPE_TYPE_RO_RECTANGLE if shape["isRotated"] == True else SHAPE_TYPE_RECTANGLE
            data_shape["group_id"] = None
            data_shape["flags"] = {}
            data_shapes.append(data_shape)

        data = {
            "labeler": __appname__,
            "label_format_version": self.LABEL_FORMAT_VERSION,
            "task_type": self._task_type,
            "shapes": data_shapes,
            "imageHeight": image_info['height'],
            "imageWidth": image_info['width'],
            "imageDepth": image_info['depth'],
            "need_confirm": True if image_info['isNeedConfirm'] == True else False,
            "split": self._split
        }
        try:
            with open(self._label_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.labelPath = self._label_path
            self._saved = True
        except Exception as e:
            raise LabelFileError(e)

    def set_shapes(self, shapes: list):
        self._shapes = shapes

    def get_shapes(self) -> List[Shape]:
        return self._shapes

    def get_is_need_confirm(self):
        return self._is_need_confirm

