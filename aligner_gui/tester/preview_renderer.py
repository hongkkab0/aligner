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
            if show_gt or show_prediction:
                self._draw_legend(painter, class_index, image.shape[1], image.shape[0], font_size)
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

    # Visually distinct color palette (R, G, B) — cycles if > 18 classes
    _PALETTE = [
        (255,  70,  70),   # 0  red
        ( 70, 160, 255),   # 1  blue
        ( 70, 210,  70),   # 2  green
        (255, 200,  40),   # 3  yellow
        (180,  70, 255),   # 4  purple
        (255, 140,  40),   # 5  orange
        ( 50, 210, 190),   # 6  teal
        (255, 100, 190),   # 7  pink
        (150, 210,  60),   # 8  lime
        (130, 130, 255),   # 9  lavender
        (255,  70, 140),   # 10 rose
        ( 50, 180, 130),   # 11 seafoam
        (230, 180,  70),   # 12 gold
        (210,  80, 110),   # 13 crimson
        ( 90, 200, 255),   # 14 sky blue
        (210, 140,  60),   # 15 caramel
        (160, 255, 130),   # 16 mint
        (255, 130,  70),   # 17 coral
    ]

    def _get_color(self, idx: int) -> tuple:
        return self._PALETTE[idx % len(self._PALETTE)]

    def _draw_gt_overlay(self, painter: QPainter, img_path: str, class_index: dict, thickness: int, font_size: int):
        label = self._get_label_data(img_path)
        if label is None:
            return

        no_rotation = self._settings_provider().no_rotation
        painter.setFont(QFont("Arial", font_size))
        for shape in label.get("shapes", []):
            class_name = shape.get("label", "")
            if class_name not in class_index:
                continue
            class_idx = class_index[class_name]
            r, g, b = self._get_color(class_idx)
            gt_pen  = QPen(QColor(r, g, b), thickness, Qt.DashLine)
            gt_fill = QColor(r, g, b, 24)
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
            r, g, b = self._get_color(class_idx)
            pred_pen  = QPen(QColor(r, g, b), thickness, Qt.SolidLine)
            pred_fill = QColor(r, g, b, 32)
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

    def _draw_legend(self, painter: QPainter, class_index: dict, image_w: int, image_h: int, font_size: int):
        """Draw a color-coded class legend in the top-left corner."""
        if not class_index:
            return

        classes = sorted(class_index.items(), key=lambda kv: kv[1])   # [(name, idx), ...]
        n = len(classes)

        margin    = max(6, font_size // 2)
        swatch_w  = max(16, font_size)
        swatch_h  = max(12, font_size - 2)
        text_pad  = 6
        row_h     = swatch_h + 4
        header_h  = font_size + 6          # room for "━ Pred  -- GT" header

        # Measure widest class name to size the background rect
        painter.setFont(QFont("Arial", font_size - 2))
        fm = painter.fontMetrics()
        max_text_w = max(fm.width(name) for name, _ in classes)
        box_w = swatch_w + text_pad + max_text_w + margin * 2
        box_h = header_h + n * row_h + margin * 2

        # Clamp to image bounds
        x0 = margin
        y0 = margin
        box_w = min(box_w, image_w - 2 * margin)
        box_h = min(box_h, image_h - 2 * margin)

        # Semi-transparent background
        painter.save()
        painter.setOpacity(0.72)
        painter.setBrush(QColor(20, 20, 20))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x0, y0, box_w, box_h, 4, 4)
        painter.setOpacity(1.0)

        # Header: style guide
        header_font = QFont("Arial", font_size - 3)
        painter.setFont(header_font)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(x0 + margin, y0 + margin + font_size - 4,
                         "━ Pred    - - GT")

        # One row per class
        for row, (class_name, class_idx) in enumerate(classes):
            r, g, b = self._get_color(class_idx)
            ry = y0 + margin + header_h + row * row_h
            sx = x0 + margin
            # Solid swatch (Pred style)
            painter.setPen(QPen(QColor(r, g, b), 2, Qt.SolidLine))
            painter.setBrush(QColor(r, g, b, 80))
            painter.drawRect(sx, ry, swatch_w, swatch_h)
            # Class name
            painter.setFont(QFont("Arial", font_size - 2))
            painter.setPen(QColor(r, g, b))
            painter.drawText(sx + swatch_w + text_pad,
                             ry + swatch_h - 2,
                             class_name)

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

