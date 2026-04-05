from __future__ import annotations

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import copy
import logging
import traceback

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_CROSS = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor
CURSOR_SIZE = Qt.SizeAllCursor
CURSOR_SIZE_VER = Qt.SizeVerCursor
CURSOR_SIZE_HOR = Qt.SizeHorCursor
CURSOR_SIZE_BDIAG = Qt.SizeBDiagCursor
CURSOR_SIZE_FDIAG = Qt.SizeFDiagCursor


class ImagePanel(QLabel):
    show_area_changed = pyqtSignal(QRect)
    roi_changed = pyqtSignal(int, int, QRect)
    roi_added = pyqtSignal(int, int, QRect)
    roi_removed = pyqtSignal(int, int)

    def resizeEvent(self, *args, **kwargs):
        super().resizeEvent(*args, **kwargs)
        if not self.image.isNull() and self._auto_fit:
            self.fitImagetoWindow()
        area = self.getShowArea()
        if area is not None:
            self.show_area_changed.emit(self.getShowArea())

    def __init__(self, widget):
        super().__init__(widget)
        self.image = QPixmap()
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._last_offset = QPoint(0, 0)
        self._pressed_point = QPoint(0, 0)
        self.roi_table = dict()
        self.roi_class_names = dict()
        self.cur_roi_class = 0  # ROI 異붽? ?????ъ슜?섎뒗 ROI Class No. ?몃??먯꽌 吏??
        self.sel_roi_class = 0  # 留덉슦???대┃ ?깆쑝濡??좏깮??ROI??Class No
        self.sel_roi_no = -1
        self.sel_roi = QRect()
        self.roi_clicked_pos = 0
        self.add_new_ROI = False
        self.roi_mode = True
        self._auto_fit = True
        self._is_panning = False
        self._pan_button = Qt.NoButton
        self.color_list = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0),
                           QColor(0, 255, 255), QColor(218, 112, 214), QColor(255, 140, 0), QColor(255, 105, 180),
                           QColor(255, 218, 185), QColor(255, 215, 0), QColor(165, 42, 42), QColor(219, 112, 147),
                           QColor(245, 245, 220), QColor(128, 128, 0), QColor(222, 184, 135), QColor(250, 128, 114)]

        self.setMouseTracking(True)

    def paintEvent(self, event):
        frame_rect = self.rect()
        painter = QPainter(self)

        if not self.image.isNull():
            painter.scale(self._scale, self._scale)
            painter.translate(self._offset)
            painter.drawPixmap(0, 0, self.image)

            """
            painter.setOpacity(0.6)
            tmpPixmap = self.image.scaled(100, 100, Qt.KeepAspectRatio)
            painter.drawPixmap(0, 0, tmpPixmap)
            """

            if self.roi_mode:
                self._drawROIs(painter)

    def _drawROIs(self, painter):
        color_no = 0
        for roi_class_no in self.roi_table:
            roi_no = 0
            for roi in self.roi_table[roi_class_no]:
                painter.setPen(QPen(self.color_list[color_no % 16], 2, Qt.DashLine))
                painter.drawRect(roi)

                font = painter.font()
                font.setPixelSize(10)
                painter.setFont(font)
                roi_class_name = str(roi_class_no)
                if roi_class_no in self.roi_class_names:
                    roi_class_name = self.roi_class_names[roi_class_no]
                painter.setPen(QPen(self.color_list[color_no % 16], 1, Qt.SolidLine))
                painter.drawText(roi.translated(2, 2), Qt.AlignLeading, roi_class_name + "_%d" % roi_no)
                painter.drawText(roi.translated(2, 17), Qt.AlignLeading,
                                 "%d,%d,%d,%d" % (roi.left(), roi.top(), roi.width(), roi.height()))

                roi_no += 1
            color_no += 1

    def wheelEvent(self, event):
        if self.image.isNull():
            return

        self._auto_fit = False
        angle = event.angleDelta()
        oldPos = self._convertToImageCoordinate(event.pos())

        if angle.y() > 0.0:
            self._scale = self._scale * 1.1
            newPos = self._convertToImageCoordinate(event.pos())
            self._offset = self._offset - oldPos + newPos
        else:
            if self._isImageSmallerThanFrame():
                self.fitImagetoWindow()
            else:
                self._scale = self._scale / 1.1
                newPos = self._convertToImageCoordinate(event.pos())
                self._offset = self._offset - oldPos + newPos
        #self._adjustOffset()
        self.update()

        self.show_area_changed.emit(self.getShowArea())

    def _isImageSmallerThanFrame(self):
        image_size = self.image.size()
        frame_size = self.size()
        scaled_image_size = image_size * self._scale

        if scaled_image_size.width() <= frame_size.width() and scaled_image_size.height() <= frame_size.height():
            return True

        return False

    def _adjustOffset(self):
        image_size = self.image.size()
        frame_size = self.size()
        scaled_image_size = image_size * self._scale

        cur_view_w = (image_size.width() + self._offset.x()) * self._scale
        cur_view_h = (image_size.height() + self._offset.y()) * self._scale

        # ?ㅼ??쇰맂 ?대?吏??蹂댁씠???곸뿭??肄섑듃濡?蹂대떎 ???뱀? ?믪씠媛 ?묒? 寃쎌슦 利??덈Т 留롮씠 ?대룞?댁꽌 ?대?吏 ?앹씠 ?덉쑝濡?紐삳뱾?댁삤寃?
        if frame_size.width() >= cur_view_w:
            tmp_offset_x = frame_size.width() / self._scale - image_size.width()
            self._offset.setX(tmp_offset_x)
        if frame_size.height() >= cur_view_h:
            tmp_offset_y = frame_size.height() / self._scale - image_size.height()
            self._offset.setY(tmp_offset_y)

        # ?대?吏 醫뚯긽??肄섑듃濡??대?濡??ㅼ뼱?ㅼ? ?딅룄濡?+Offset??紐살＜寃??섍퀬
        # ?ㅼ??쇰맂 ?대?吏????씠???믪씠媛 ?꾨젅??蹂대떎 ?묒쑝硫?Offset??紐살＜寃??쒕떎.
        if self._offset.x() > 0 or scaled_image_size.width() < frame_size.width():
            self._offset.setX(0)
        if self._offset.y() > 0 or scaled_image_size.height() < frame_size.height():
            self._offset.setY(0)

    def mouseMoveEvent(self, event):
        if self.image.isNull():
            return

        cur_pos = event.pos()
        cur_btn = event.buttons()

        if self._is_panning and (cur_btn & self._pan_button):
            self._auto_fit = False
            dist = (cur_pos - self._pressed_point) / self._scale
            self._offset = self._last_offset + dist
            self._adjustOffset()
            self.update()
            self.show_area_changed.emit(self.getShowArea())
        elif cur_btn & Qt.LeftButton:
            if self.roi_mode:
                if self.sel_roi_no >= 0:
                    self._auto_fit = False
                    dist = (cur_pos - self._pressed_point) / self._scale
                    if self.roi_clicked_pos == 0:
                        self.roi_table[self.sel_roi_class][self.sel_roi_no] = self.sel_roi.translated(dist)
                    if self.roi_clicked_pos & 1:
                        self.roi_table[self.sel_roi_class][self.sel_roi_no].setTop(self.sel_roi.top() + dist.y())
                    if self.roi_clicked_pos & 2:
                        self.roi_table[self.sel_roi_class][self.sel_roi_no].setBottom(self.sel_roi.bottom() + dist.y())
                    if self.roi_clicked_pos & 4:
                        self.roi_table[self.sel_roi_class][self.sel_roi_no].setLeft(self.sel_roi.left() + dist.x())
                    if self.roi_clicked_pos & 8:
                        self.roi_table[self.sel_roi_class][self.sel_roi_no].setRight(self.sel_roi.right() + dist.x())

                    self._adjustROI(self.roi_table[self.sel_roi_class][self.sel_roi_no], self.image.rect())
                    self.update()
        else:
            self._changeCursor(cur_pos, event)

    def _changeCursor(self, pos, event):
        roi_class, roi_no, roi_pos = self._roiHitTest(pos)
        if roi_no >= 0:
            if roi_pos == 1 or roi_pos == 2:
                self._overrideCursor(CURSOR_SIZE_VER)
            elif roi_pos == 4 or roi_pos == 8:
                self._overrideCursor(CURSOR_SIZE_HOR)
            elif roi_pos == 5 or roi_pos == 10:
                self._overrideCursor(CURSOR_SIZE_FDIAG)
            elif roi_pos == 6 or roi_pos == 9:
                self._overrideCursor(CURSOR_SIZE_BDIAG)
            elif roi_pos == 0 and not roi_no == -1:
                self._overrideCursor(CURSOR_SIZE)
        else:
            self._restoreCursor()

    def _adjustROI(self, roi, bound):
        if roi.left() < 0:
            roi.setLeft(0)
        elif roi.left() >= bound.width():
            roi.setLeft(bound.width() - 1)

        if roi.right() < 0:
            roi.setRight(0)
        elif roi.right() >= bound.width():
            roi.setRight(bound.width() - 1)

        if roi.top() < 0:
            roi.setTop(0)
        elif roi.top() >= bound.height():
            roi.setTop(bound.height() - 1)

        if roi.bottom() < 0:
            roi.setBottom(0)
        elif roi.bottom() >= bound.height():
            roi.setBottom(bound.height() - 1)

    def mousePressEvent(self, event):
        if self.image.isNull():
            return

        cur_pos = event.pos()
        cur_btn = event.buttons()
        self._pressed_point = cur_pos

        if cur_btn & Qt.LeftButton:
            if self.roi_mode:
                if event.modifiers() & Qt.ControlModifier:
                    left_top = self._convertToImageCoordinate(cur_pos)
                    self.sel_roi = QRect(left_top, QSize(0, 0))
                    self.addROI(self.cur_roi_class, QRect(left_top, QSize(0, 0)))
                    self.sel_roi_no = len(self.roi_table[self.cur_roi_class]) - 1
                    self.sel_roi_class = self.cur_roi_class
                    self.roi_clicked_pos = 10
                    self._overrideCursor(CURSOR_CROSS)
                    self.add_new_ROI = True
                else:
                    self.sel_roi_class, self.sel_roi_no, self.roi_clicked_pos = self._roiHitTest(cur_pos)
                    if self.sel_roi_no >= 0:
                        self.sel_roi = copy.deepcopy(self.roi_table[self.sel_roi_class][self.sel_roi_no])
                        self._changeCursor(cur_pos, event)
                    else:
                        self._last_offset = self._offset
                        self._auto_fit = False
                        self._is_panning = True
                        self._pan_button = Qt.LeftButton
                        self._overrideCursor(CURSOR_MOVE)
            else:
                self._last_offset = self._offset
                self._auto_fit = False
                self._is_panning = True
                self._pan_button = Qt.LeftButton
                self._overrideCursor(CURSOR_MOVE)
        elif cur_btn & Qt.RightButton:
            self._last_offset = self._offset
            self._auto_fit = False
            self._is_panning = True
            self._pan_button = Qt.RightButton
            self._overrideCursor(CURSOR_MOVE)

    def mouseReleaseEvent(self, event):
        if self.image.isNull():
            return

        self._is_panning = False
        self._pan_button = Qt.NoButton

        if not self.roi_mode:
            self._restoreCursor()
            return

        if 0 <= self.sel_roi_no < len(self.roi_table[self.sel_roi_class]):
            if not self.roi_table[self.sel_roi_class][self.sel_roi_no].isValid():
                self.roi_table[self.sel_roi_class][self.sel_roi_no] = self.roi_table[self.sel_roi_class][self.sel_roi_no].normalized()
                self.update()

            if self.roi_table[self.sel_roi_class][self.sel_roi_no].width() < 10 or self.roi_table[self.sel_roi_class][self.sel_roi_no].height() < 10:
                if self.add_new_ROI:
                    self.add_new_ROI = False
                else:
                    self.roi_removed.emit(self.sel_roi_class, self.sel_roi_no)

                self.roi_table[self.sel_roi_class].pop(self.sel_roi_no)
                self.sel_roi_no = -1
                self.update()
            else:
                if self.add_new_ROI:
                    self.add_new_ROI = False
                    self.roi_added.emit(self.sel_roi_class, self.sel_roi_no, self.roi_table[self.sel_roi_class][self.sel_roi_no])
                else:
                    self.roi_changed.emit(self.sel_roi_class, self.sel_roi_no, self.roi_table[self.sel_roi_class][self.sel_roi_no])

        self._restoreCursor()

    def _convertToImageCoordinate(self, point):
        image_point = point / self._scale - self._offset
        return image_point

    def _roiHitTest(self, point):
        roi_clicked_pos = 0
        margin = 3
        for roi_class in self.roi_table:
            roi_index = 0
            for roi in self.roi_table[roi_class]:
                image_point = self._convertToImageCoordinate(point)
                roi_margin = roi.marginsAdded(QMargins(1, 1, 2, 2))
                if roi_margin.contains(image_point):
                    if margin > abs(image_point.y() - roi.top()):
                        roi_clicked_pos = roi_clicked_pos | 1
                    elif margin > abs(roi.bottom() - image_point.y()):
                        roi_clicked_pos = roi_clicked_pos | 2

                    if margin > abs(image_point.x() - roi.left()):
                        roi_clicked_pos = roi_clicked_pos | 4
                    elif margin > abs(roi.right() - image_point.x()):
                        roi_clicked_pos = roi_clicked_pos | 8

                    return roi_class, roi_index, roi_clicked_pos

                roi_index = roi_index + 1

        return 0, -1, 0

    def _currentCursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def _overrideCursor(self, cursor):
        if self._currentCursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def _restoreCursor(self):
        QApplication.restoreOverrideCursor()

    def setImage(self, pixmap):
        try:
            image_size = pixmap.size()
            frame_size = self.size()

            if image_size.width() == 0 or image_size.height() == 0:
                return

            tmp_scale_x = frame_size.width() / image_size.width()
            tmp_scale_y = frame_size.height() / image_size.height()

            if tmp_scale_x < tmp_scale_y:
                self._scale = tmp_scale_x
            else:
                self._scale = tmp_scale_y

            self._offset = QPoint(0, 0)
            self._last_offset = QPoint(0, 0)
            self._pressed_point = QPoint(0, 0)
            self.roi_table = dict()
            self.cur_roi_class = 0
            self.sel_roi_class = 0
            self.sel_roi_no = -1
            self.sel_roi = QRect()
            self.roi_clicked_pos = 0
            self.image = pixmap
            self._auto_fit = True
            self._is_panning = False
            self._pan_button = Qt.NoButton
            self.update()
        except Exception as e:
            print(e)

    def setWorkingROIClassName(self, roi_class_no):
        self.cur_roi_class = roi_class_no

    def addROI(self, roi_class_no, roi):
        if roi_class_no in self.roi_table:  # ?대? 議댁옱?섎뒗 ?대옒?ㅼ뿉 ROI瑜?異붽??섎㈃
            self.roi_table[roi_class_no].append(roi)
        else:
            self.roi_table[roi_class_no] = [roi]

    def removeROI(self, roi_class_no, roi_no):
        if roi_class_no in self.roi_table:
            if 0 <= roi_no < len(self.roi_table[roi_class_no]):
                self.roi_table[roi_class_no].pop(roi_no)

    def setROIList(self, roi_class_no, new_roi_list):
        self.roi_table[roi_class_no] = new_roi_list

    def getROIList(self, roi_class_no):
        if roi_class_no in self.roi_table:
            return self.roi_table[roi_class_no]

        return None

    def getROI(self, roi_class_no, roi_no):
        if roi_class_no in self.roi_table:
            if 0 <= roi_no < len(self.roi_table[roi_class_no]):
                return self.roi_table[roi_class_no][roi_no]

        return None

    def setROIColor(self, roi_class_no, roi_color):
        if roi_class_no < 0:
            return
        self.color_list[roi_class_no % 16] = roi_color

    def getShowArea(self):
        if self.image.isNull():
            return

        left_top = self._offset * -1
        rect_size = self.size() / self._scale

        if rect_size.width() > self.image.width():
            rect_size.setWidth(self.image.width())
        if rect_size.height() > self.image.height():
            rect_size.setHeight(self.image.height())

        return QRect(left_top, rect_size)

    def setROIMode(self, enable):
        self.roi_mode = enable

    def setROIClassName(self, roi_class_no, roi_class_name):
        self.roi_class_names[roi_class_no] = roi_class_name

    def saveImage(self, save_path):
        try:
            self.image.save(save_path)
            logging.info("Image was saved successfully at " + save_path)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            error_msg = "ERROR - " + str(e)
            logging.error(error_msg)

    def fitImagetoWindow(self):
        tmp_scale_x = self.size().width() / self.image.width()
        tmp_scale_y = self.size().height() / self.image.height()
        tmp_scale = tmp_scale_x
        if tmp_scale_x > tmp_scale_y:
            tmp_scale = tmp_scale_y
        self._offset = QPoint(0, 0)
        self._scale = tmp_scale
