from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Callable

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF

from aligner_engine.mm_rotate_det.dice.remove_rotation import remove_rotation
from aligner_engine.summary import ResultSummary
from aligner_gui.shared import gui_util
from aligner_gui.shared.image_cache import CachedImageReader, decode_image_with_cv2


class TesterPreviewRenderer:
    def __init__(
        self,
        cv2_provider: Callable[[], object],
        settings_provider: Callable[[], object],
        image_cache_size: int = 12,
        preview_cache_size: int = 24,
        label_cache_size: int = 32,
    ):
        self._cv2_provider = cv2_provider
        self._settings_provider = settings_provider
        self._image_reader = CachedImageReader(
            decoder=lambda image_path: decode_image_with_cv2(image_path, self._cv2_provider()),
            max_items=image_cache_size,
        )
        self._preview_cache: OrderedDict[tuple, QPixmap] = OrderedDict()
        self._label_cache: OrderedDict[str, object] = OrderedDict()
        self._preview_cache_size = max(1, int(preview_cache_size))
        self._label_cache_size = max(1, int(label_cache_size))
        self._colormap = None

    def clear(self):
        self._image_reader.clear()
        self._preview_cache.clear()
        self._label_cache.clear()

    def render_file_preview(self, img_path: str):
        cache_key = ("file", os.path.abspath(img_path))
        cached = self._get_cached_preview(cache_key)
        if cached is not None:
            return cached

        image = self._image_reader.read(img_path)
        if image is None:
            return None
        pixmap = gui_util.cvimg_to_pixmap(image)
        if pixmap is None or pixmap.isNull():
            return None
        self._put_preview(cache_key, pixmap)
        return QPixmap(pixmap)

    def render_detail_preview(
        self,
        img_path: str,
        result_summary: ResultSummary,
        class_index: dict,
        show_gt: bool,
        show_prediction: bool,
    ):
        cache_key = ("detail", os.path.abspath(img_path), bool(show_gt), bool(show_prediction))
        cached = self._get_cached_preview(cache_key)
        if cached is not None:
            return cached

        image = self._image_reader.read(img_path)
        if image is None:
            return None

        pixmap = gui_util.cvimg_to_pixmap(image)
        if pixmap is None or pixmap.isNull():
            return None

        painter = QPainter()
        if not painter.begin(pixmap):
            return None

        try:
            thickness, font_size = self._get_overlay_style(image)
            if show_gt:
                self._draw_gt_overlay(painter, img_path, class_index, thickness, font_size)
            if show_prediction:
                self._draw_prediction_overlay(
                    painter,
                    img_path,
                    result_summary,
                    class_index,
                    thickness,
                    font_size,
                )
        finally:
            painter.end()

        self._put_preview(cache_key, pixmap)
        return QPixmap(pixmap)

    def _get_overlay_style(self, image):
        img_h = image.shape[0]
        img_w = image.shape[1]
        long_axis = max(img_h, img_w)
        if long_axis < 128:
            return 1, 12
        if long_axis < 256:
            return 2, 14
        if long_axis < 512:
            return 3, 16
        if long_axis < 1024:
            return 4, 18
        if long_axis < 2048:
            return 5, 20
        return 6, 20

    def _draw_gt_overlay(self, painter: QPainter, img_path: str, class_index: dict, thickness: int, font_size: int):
        label = self._get_label_data(img_path)
        if label is None:
            return

        no_rotation = self._settings_provider().no_rotation
        painter.setFont(QFont("Arial", font_size))
        gt_pen = QPen(QColor(0, 255, 120), thickness, Qt.DashLine)
        gt_fill = QColor(0, 255, 120, 24)
        for shape in label.get("shapes", []):
            class_name = shape.get("label", "")
            if class_name not in class_index:
                continue
            qbox = [
                shape["x1"], shape["y1"],
                shape["x2"], shape["y2"],
                shape["x3"], shape["y3"],
                shape["x4"], shape["y4"],
            ]
            if no_rotation:
                qbox = remove_rotation(qbox)
            painter.save()
            painter.setBrush(gt_fill)
            self._draw_qbox(painter, qbox, f"GT:{class_name}", gt_pen, text_offset=-4)
            painter.restore()

    def _draw_prediction_overlay(
        self,
        painter: QPainter,
        img_path: str,
        result_summary: ResultSummary,
        class_index: dict,
        thickness: int,
        font_size: int,
    ):
        data_result = result_summary.data_result.get(img_path)
        if not data_result:
            return

        painter.setFont(QFont("Arial", font_size))
        for _, detection in data_result.items():
            class_name = detection["class_name"]
            if class_name not in class_index:
                continue
            class_idx = class_index[class_name]
            color = self._get_color(class_idx)
            pred_pen = QPen(QColor(color[0], color[1], color[2]), thickness)
            pred_fill = QColor(color[0], color[1], color[2], 32)
            painter.save()
            painter.setBrush(pred_fill)
            self._draw_qbox(
                painter,
                detection["qbox"],
                f'{class_name} ({detection["conf"]:.2f})',
                pred_pen,
                text_offset=0,
            )
            painter.restore()

    def _draw_qbox(self, painter: QPainter, qbox, text: str, pen: QPen, text_offset: int = -4):
        painter.setPen(pen)
        polygon = QPolygonF([
            QPointF(qbox[0], qbox[1]),
            QPointF(qbox[2], qbox[3]),
            QPointF(qbox[4], qbox[5]),
            QPointF(qbox[6], qbox[7]),
        ])
        painter.drawPolygon(polygon)
        if text:
            painter.drawText(qbox[0], qbox[1] + text_offset, text)

    def _get_color(self, idx: int):
        if self._colormap is None:
            import imgviz

            self._colormap = imgviz.label_colormap(value=1.5)
        return self._colormap[(idx + 3) % len(self._colormap)]

    def _get_label_data(self, img_path: str):
        normalized_path = os.path.abspath(img_path)
        cached = self._label_cache.get(normalized_path)
        if cached is not None:
            self._label_cache.move_to_end(normalized_path)
            return cached

        label_path = os.path.splitext(normalized_path)[0] + ".json"
        if not os.path.exists(label_path):
            return None

        try:
            with open(label_path, encoding="utf-8") as f:
                label = json.load(f)
        except Exception:
            return None

        self._label_cache[normalized_path] = label
        self._label_cache.move_to_end(normalized_path)
        while len(self._label_cache) > self._label_cache_size:
            self._label_cache.popitem(last=False)
        return label

    def _get_cached_preview(self, cache_key):
        cached = self._preview_cache.get(cache_key)
        if cached is None:
            return None
        self._preview_cache.move_to_end(cache_key)
        return QPixmap(cached)

    def _put_preview(self, cache_key, pixmap: QPixmap):
        self._preview_cache[cache_key] = QPixmap(pixmap)
        self._preview_cache.move_to_end(cache_key)
        while len(self._preview_cache) > self._preview_cache_size:
            self._preview_cache.popitem(last=False)

