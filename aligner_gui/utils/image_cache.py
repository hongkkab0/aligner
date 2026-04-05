from __future__ import annotations

import copy
import os
from collections import OrderedDict
from typing import Callable

import numpy as np


def decode_image_with_cv2(image_path: str, cv2_module):
    try:
        with open(image_path, "rb") as fp:
            buffer = np.fromfile(fp, np.uint8)
            if buffer.size == 0:
                return None
            return cv2_module.imdecode(buffer, cv2_module.IMREAD_COLOR)
    except IOError:
        return None


class CachedImageReader:
    def __init__(self, decoder: Callable[[str], object], max_items: int = 12):
        self._decoder = decoder
        self._max_items = max(1, int(max_items))
        self._cache: OrderedDict[str, object] = OrderedDict()

    def clear(self):
        self._cache.clear()

    def read(self, image_path: str):
        normalized_path = os.path.abspath(image_path)
        cached = self._cache.get(normalized_path)
        if cached is not None:
            self._cache.move_to_end(normalized_path)
            return copy.copy(cached)

        image = self._decoder(normalized_path)
        if image is None:
            return None

        cached_image = copy.copy(image)
        self._cache[normalized_path] = cached_image
        self._cache.move_to_end(normalized_path)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)
        return copy.copy(cached_image)
