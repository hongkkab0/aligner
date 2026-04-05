from typing import List, Set, Dict, Tuple
from copy import deepcopy
import imgviz


class LabelManager:
    _label_names_with_idx: Dict[str, int] = {}  # This have label names and idxex, idx starts from 0
    LABEL_COLORMAP = imgviz.label_colormap(value=1.5)

    @classmethod
    def init(cls):
        cls._label_names_with_idx = {}

    @classmethod
    def update_label_names_with_idx(cls, label_name):
        if label_name=="":
            return
        if label_name not in cls._label_names_with_idx:
            cls._label_names_with_idx[label_name] = len(cls._label_names_with_idx)

    @classmethod
    def get_label_names_with_idx(cls):
        return deepcopy(cls._label_names_with_idx)

    @classmethod
    def get_rgb_by_label(cls, label):
        cls.update_label_names_with_idx(label)
        label_idx = cls._label_names_with_idx[label]
        return cls.LABEL_COLORMAP[(label_idx + 1) % len(cls.LABEL_COLORMAP)]