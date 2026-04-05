#!/usr/bin/python
# -*- coding: utf-8 -*-



from PyQt5.QtGui import *
from PyQt5.QtCore import *


from aligner_gui.labeler.libs.lib import distance
from aligner_gui.labeler.libs.label_manager import LabelManager
import math

DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)
POINT_EQUAL_TOLERANCE = 0.001

SHAPE_TYPE_RO_RECTANGLE = "ro_rectangle"
SHAPE_TYPE_RECTANGLE = "rectangle"

class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label="", line_color=None,difficult = False):
        self._label: str = ""
        self.set_label(label)
        self.points = []
        self.fill = False
        self.selected = False
        self.difficult = difficult

        self.direction = 0  # added by hy
        self.center = None # added by hy
        self.isRotated = True 

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False
        self._vertex_fill_color_to_draw = DEFAULT_VERTEX_FILL_COLOR

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color


    def rotate(self, theta):
        for i, p in enumerate(self.points):
            self.points[i] = self.rotatePoint(p, theta)
        self.direction -= theta
        self.direction = self.direction % (2 * math.pi)

    def rotatePoint(self, p, theta):
        order = p-self.center;
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = - sinTheta * order.x() + cosTheta * order.y()
        pRes = QPointF(self.center.x() + pResx, self.center.y() + pResy)
        return pRes

    def close(self):
        self.center = QPointF((self.points[0].x()+self.points[2].x()) / 2, (self.points[0].y()+self.points[2].y()) / 2)
        # print("refresh center!")
        self._closed = True

    def reachMaxPoints(self):
        if len(self.points) >= 4:
            return True
        return False

    def addPoint(self, point):
        if self.points and len(self.points) == 4 and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        if self.points:
            color = self.select_line_color if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            line_path.moveTo(self.points[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            #self.drawVertex(vrtx_path, 0)

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                # print('shape paint points (%d, %d)' % (p.x(), p.y()))
                self.drawVertex(vrtx_path, i)
            if self.isClosed():
                line_path.lineTo(self.points[0])

            # dir_path = QPainterPath()
            # tempP = self.points[0]+QPointF(10,10)
            # print('direction2 is %lf, a os %lf' % (self.direction,math.tan(self.direction)))
            # dir_path.moveTo(tempP)
            # dir_path.lineTo(tempP + QPointF(10, (10-tempP.x())* math.tan(self.direction)+tempP.y()))
            # painter.drawPath(dir_path)

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self._vertex_fill_color_to_draw)
            if self.fill:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            if self.center is not None:
                center_path = QPainterPath()
                d = self.point_size / self.scale
                center_path.addRect(self.center.x() - d / 2, self.center.y() - d / 2, d, d)
                painter.drawPath(center_path)
                if self.isRotated:
                    painter.fillPath(center_path, self._vertex_fill_color_to_draw)
                else:
                    painter.fillPath(center_path, QColor(0, 0, 0))

    def paintNormalCenter(self, painter):
        if self.center is not None:
            center_path = QPainterPath();
            d = self.point_size / self.scale
            center_path.addRect(self.center.x() - d / 2, self.center.y() - d / 2, d, d)
            painter.drawPath(center_path)
            if not self.isRotated:
                painter.fillPath(center_path, QColor(0, 0, 0))

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color_to_draw = self.hvertex_fill_color
        else:
            self._vertex_fill_color_to_draw = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"
    # def drawVertex(self, path, center):
    #     pass

    def nearestVertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            if distance(p - point) <= epsilon:
                return i
        return None

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def bounding_rect(self):
        return self.makePath().boundingRect()

    def move_by(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        shape = Shape("%s" % self._label)
        shape.points = [p for p in self.points]
        
        shape.center = self.center
        shape.direction = self.direction
        shape.isRotated = self.isRotated

        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult 
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    def get_label(self):
        return self._label

    def set_label(self, label):
        self._label = label
        if label == "":
            return
        rgb = LabelManager.get_rgb_by_label(label)
        r, g, b = rgb
        self.line_color = QColor(r, g, b)
        self.vertex_fill_color = QColor(r, g, b)
        self.hvertex_fill_color = QColor(255, 255, 255)
        self.fill_color = QColor(r, g, b, 30)
        self.select_line_color = QColor(255, 255, 255)
        self.select_fill_color = QColor(r, g, b, 50)

    def change_longside_shortside_of_shape_box(self, long, short):
        direction = self.direction
        self.direction = 0
        center_buff = self.center

        point_0 = [self.points[0].x(), self.points[0].y()]
        point_1 = [self.points[1].x(), self.points[1].y()]
        point_2 = [self.points[2].x(), self.points[2].y()]

        side_0 = math.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
        side_1 = math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

        if side_1 > side_0:
            if (long - POINT_EQUAL_TOLERANCE < side_1 < long + POINT_EQUAL_TOLERANCE
                    and short - POINT_EQUAL_TOLERANCE < side_0 < short + POINT_EQUAL_TOLERANCE):
                return False
        else:
            if (long - POINT_EQUAL_TOLERANCE < side_0 < long + POINT_EQUAL_TOLERANCE
                    and short - POINT_EQUAL_TOLERANCE < side_1 < short + POINT_EQUAL_TOLERANCE):
                return False

        if side_0 >= side_1:
            self.points[0] = QPointF(0.0, 0.0)
            self.points[1] = QPointF(long, 0.0)
            self.points[2] = QPointF(long, short)
            self.points[3] = QPointF(0.0, short)
        else:
            self.points[0] = QPointF(0.0, 0.0)
            self.points[1] = QPointF(short, 0.0)
            self.points[2] = QPointF(short, long)
            self.points[3] = QPointF(0.0, long)

        self.close()
        center_offset = center_buff - self.center
        self.move_by(center_offset)
        self.close()

        self.rotate(-direction)

        return True







