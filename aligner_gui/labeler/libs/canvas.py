from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from aligner_gui.labeler.libs.shape import Shape
from aligner_gui.labeler.libs.lib import distance
import math
from typing import List, Set, Dict, Tuple
from copy import deepcopy

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE_SHAPE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor
CURSOR_MOVE_VIEW_POINT = Qt.ClosedHandCursor
SHORTCUT_MOVE_VIEW_POINT = Qt.ControlModifier

class Canvas(QWidget):
    sig_zoom_request = pyqtSignal(int, QPoint)
    sig_scroll_request = pyqtSignal(int, int)
    sig_new_shape_made = pyqtSignal()
    moveViewPointHorizontalRequest = pyqtSignal(int)
    moveViewPointVerticalRequest = pyqtSignal(int)
    moveViewPointPressed = pyqtSignal()

    sig_shape_selection_changed = pyqtSignal(bool)
    shapeMoved = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)

    hideRRect = pyqtSignal(bool)
    hideNRect = pyqtSignal(bool)
    status = pyqtSignal(str)

    CREATE, EDIT = list(range(2))

    epsilon = 11.0



    def __init__(self, labeler, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self._labeler = labeler
        self.mode = self.EDIT
        self._shapes: List[Shape] = []
        self._current_shape: Shape = None
        self.selected_shape:Shape = None  # save the selected shape here
        self.selectedShapeCopy = None
        self.lineColor = QColor(0, 0, 255)
        self.line = Shape(line_color=self.lineColor)
        self.prevPoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.pixmap = QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape : Shape = None
        self.hVertex = None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        self.menus = (QMenu(),)
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        # judge can draw rotate rect
        self.canDrawRotatedRect = True
        self.hideRotated = False
        self.hideNormal = False
        self._can_out_of_bounding = False
        self.showCenter = False
        self._moving_shape_by_mouse = False
        # Ctrl + Mouse Move Value.
        self._is_moving_view_point = False
        self._view_point_pressed = QPoint(0, 0)

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hVertex = self.hShape = None

    def selectedVertex(self):
        return self.hVertex is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        pos = self.transformPos(ev.pos())

        global_mouse_pos = self.mapToGlobal(ev.pos())
        is_cursor_setted = False

        # move view point
        if self._is_moving_view_point == True:
            if self.pixmap.width() == 0:
                self._is_moving_view_point = False
                return
            if self.pixmap.height() == 0:
                self._is_moving_view_point = False
                return
            self.overrideCursor(CURSOR_MOVE_VIEW_POINT)
            delta_x = self._view_point_pressed.x() - global_mouse_pos.x()
            delta_y = self._view_point_pressed.y() - global_mouse_pos.y()
            self.moveViewPointHorizontalRequest.emit(delta_x)
            self.moveViewPointVerticalRequest.emit(delta_y)
            return


        # Polygon drawing.
        if self.drawing():

            self.overrideCursor(CURSOR_DRAW)
            is_cursor_setted = True
            if self._current_shape:
                color = self.lineColor
                if self.out_of_pixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's edges.
                    pos = self.intersectionPoint(self._current_shape[-1], pos)
                elif len(self._current_shape) > 1 and self.closeEnough(pos, self._current_shape[0]):
                    # Attract line to starting point and colorise to alert the
                    # user:                    
                    pos = self._current_shape[0]
                    color = self._current_shape.line_color
                    self.overrideCursor(CURSOR_POINT)
                    is_cursor_setted = True
                    self._current_shape.highlightVertex(0, Shape.NEAR_VERTEX)
                self.line[1] = pos
                self.line.line_color = color
                self.update()
                self._current_shape.highlightClear()
                self.status.emit("width is %d, height is %d." % (pos.x()-self.line[0].x(), pos.y()-self.line[0].y()))

            if not is_cursor_setted:  # set mouse cursor to default
                self.restoreCursor()
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            if self.selectedVertex() and self.selected_shape.isRotated:
                self.boundedRotateShape(pos)
                self._moving_shape_by_mouse = True

                self.update()
            self.status.emit("(%d,%d)." % (pos.x(), pos.y()))

            if not is_cursor_setted:  # set mouse cursor to default
                self.restoreCursor()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                # if self.outOfPixmap(pos):
                #     print("chule ")
                #     return
                # else:
                # print("meiyou chujie")
                self.boundedMoveVertex(pos)
                self._moving_shape_by_mouse = True
                self.update()
            elif self.selected_shape and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE_SHAPE)
                is_cursor_setted = True
                self.boundedMoveShape(self.selected_shape, pos)
                self._moving_shape_by_mouse = True
                self.update()
                self.status.emit("(%d,%d)." % (pos.x(), pos.y()))

            if not is_cursor_setted:  # set mouse cursor to default
                self.restoreCursor()
            return

        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip("Image")
        for shape in reversed([s for s in self._shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = index, shape
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                is_cursor_setted = True
                # self.setToolTip("Click & drag to move point.")
                # self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = None, shape
                center = self.hShape.center
                _dir = self.hShape.direction
                angle = self.hShape.direction * 180 / math.pi;
                # label = self.hShape._label
                if self.hShape is not None:
                    label = self.hShape._label
                else:
                    label = ""

                self.setToolTip("Center : {0}, {1} \nAngle : {2} \nLabel : {3}".format(round(center.x(), 1), round(center.y(), 1), round(angle, 3), label))

                # self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                is_cursor_setted = True
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape = None, None

        if not is_cursor_setted:  # set mouse cursor to default
            self.restoreCursor()
        
        self.status.emit("(%d,%d)." % (pos.x(), pos.y()))
        

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.pos())
        # print('sldkfj %d %d' % (pos.x(), pos.y()))\
        global_mouse_pos = self.mapToGlobal(ev.pos())

        if ev.button() == Qt.ForwardButton: # mouse side forward button
            self._labeler.openPrevImg()
        elif ev.button() == Qt.BackButton: # mouse side backward button
            self._labeler.openNextImg()

        # move view point
        elif ev.button() == Qt.LeftButton and (
            SHORTCUT_MOVE_VIEW_POINT & ev.modifiers()
            or (self.editing() and not self.drawing() and not self._has_hit_shape(pos))
        ):
            self.overrideCursor(CURSOR_MOVE_VIEW_POINT)
            self._is_moving_view_point = True
            self._view_point_pressed = global_mouse_pos
            self.moveViewPointPressed.emit()

        elif ev.button() == Qt.LeftButton:
            self.hideBackroundShapes(True)
            if self.drawing():
                self.handleDrawing(pos)
            else:                
                self.selectShapePoint(pos)
                self.prevPoint = pos
                self.update()
        elif ev.button() == Qt.RightButton and self.editing():
            self.selectShapePoint(pos)
            self.hideBackroundShapes(True)
            # if self.selectedShape is not None:
            #     print('point is (%d, %d)' % (pos.x(), pos.y()))
            #     self.selectedShape.rotate(10)

            self.prevPoint = pos
            self.update()

    def mouseReleaseEvent(self, ev):
        self._is_moving_view_point = False
        self.hideBackroundShapes(False)      
        if ev.button() == Qt.RightButton and not self.selectedVertex():            
            menu = self.menus[0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapeCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapeCopy = None
                self.update()
        elif ev.button() == Qt.LeftButton and self.selected_shape:
            self.overrideCursor(CURSOR_GRAB)
        elif ev.button() == Qt.LeftButton:
            pos = self.transformPos(ev.pos())
            if self.drawing():
                self.handleDrawing(pos)

        if self._moving_shape_by_mouse == True:
            self.shapeMoved.emit()
            self._moving_shape_by_mouse = False

    def endMove(self, copy=False):
        assert self.selected_shape and self.selectedShapeCopy
        shape = self.selectedShapeCopy
        #del shape.fill_color
        #del shape.line_color
        if copy:
            self._shapes.append(shape)
            self.selected_shape.selected = False
            self.selected_shape = shape
            self.update()
        else:
            self.selected_shape.points = [p for p in shape.points]
        self.selectedShapeCopy = None

    def hideBackroundShapes(self, value):
        # print("hideBackroundShapes")
        self.hideBackround = value
        if self.selected_shape:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def handleDrawing(self, pos):
        if self._current_shape and self._current_shape.reachMaxPoints() is False:
            initPos = self._current_shape[0]
            minX = initPos.x()
            minY = initPos.y()
            targetPos = self.line[1]
            maxX = targetPos.x()
            maxY = targetPos.y()
            self._current_shape.addPoint(QPointF(maxX, minY))
            self._current_shape.addPoint(targetPos)
            self._current_shape.addPoint(QPointF(minX, maxY))
            self._current_shape.addPoint(initPos)
            self.line[0] = self._current_shape[-1]
            if self._current_shape.isClosed():
                self.finalise()
        elif not self.out_of_pixmap(pos):
            self._current_shape = Shape()
            self._current_shape.addPoint(pos)
            self.line.points = [pos, pos]
            self.setHiding()
            self.drawingPolygon.emit(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self._current_shape and len(self._current_shape) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self._current_shape) > 3:
            self._current_shape.popPoint()
            self.finalise()

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selected_shape = shape
        self.setHiding()
        self.sig_shape_selection_changed.emit(True)
        self.update()

    def selectShapePoint(self, point):
        """Select the first shape created which contains this point."""
        self.deSelectShape()
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)

            shape.selected = True
            self.selected_shape = shape
            self.calculateOffsets(shape, point)
            self.setHiding()
            self.sig_shape_selection_changed.emit(True)

            return
        for shape in reversed(self._shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                shape.selected = True
                self.selected_shape = shape
                self.calculateOffsets(shape, point)
                self.setHiding()
                self.sig_shape_selection_changed.emit(True)
                return

    def calculateOffsets(self, shape, point):
        rect = shape.bounding_rect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width()) - point.x()
        y2 = (rect.y() + rect.height()) - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def _has_hit_shape(self, point):
        for shape in reversed([s for s in self._shapes if self.isVisible(s)]):
            if shape.nearestVertex(point, self.epsilon) is not None:
                return True
            if shape.containsPoint(point):
                return True
        return False

    def boundedMoveVertex(self, pos):
        # print("Moving Vertex")
        index, shape = self.hVertex, self.hShape
        point = shape[index]

        if not self._can_out_of_bounding and self.out_of_pixmap(pos):
            return
            # pos = self.intersectionPoint(point, pos)

        # print("index is %d" % index)
        sindex = (index + 2) % 4
        # get the other 3 points after transformed
        p2,p3,p4 = self.getAdjointPoints(shape.direction, shape[sindex], pos, index)
        
        pcenter = (pos+p3)/2        
        if self._can_out_of_bounding and self.out_of_pixmap(pcenter):
            return
        # if one pixal out of map , do nothing
        if not self._can_out_of_bounding and (self.out_of_pixmap(p2) or
                                              self.out_of_pixmap(p3) or
                                              self.out_of_pixmap(p4)):
                return

        # move 4 pixal one by one 
        shape.moveVertexBy(index, pos - point)
        lindex = (index + 1) % 4
        
        rindex = (index + 3) % 4
        shape[lindex] = p2
        # shape[sindex] = p3
        shape[rindex] = p4
        shape.close()

        # calculate the height and weight, and show it
        w = math.sqrt((p4.x()-p3.x()) ** 2 + (p4.y()-p3.y()) ** 2)
        h = math.sqrt((p3.x()-p2.x()) ** 2 + (p3.y()-p2.y()) ** 2)
        self.status.emit("width is %d, height is %d." % (w,h))

    
    def getAdjointPoints(self, theta, p3, p1, index):
        # p3 = center
        # p3 = 2*center-p1
        a1 = math.tan(theta)
        if (a1 == 0):
            if index % 2 == 0:
                p2 = QPointF(p3.x(), p1.y())
                p4 = QPointF(p1.x(), p3.y())
            else:            
                p4 = QPointF(p3.x(), p1.y())
                p2 = QPointF(p1.x(), p3.y())
        else:    
            a3 = a1
            a2 = - 1/a1
            a4 = - 1/a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.getCrossPoint(a1,b1,a4,b4)
                p4 = self.getCrossPoint(a2,b2,a3,b3)
            else:            
                p4 = self.getCrossPoint(a1,b1,a4,b4)
                p2 = self.getCrossPoint(a2,b2,a3,b3)

        return p2,p3,p4

    def getCrossPoint(self,a1,b1,a2,b2):
        x = (b2-b1)/(a1-a2)
        y = (a1*b2 - a2*b1)/(a1-a2)
        return QPointF(x,y)

    def boundedRotateShape(self, pos):
        # print("Rotate Shape2")          
        # judge if some vertex is out of pixma
        index, shape = self.hVertex, self.hShape
        point = shape[index]

        angle = self.getAngle(shape.center,pos,point)
        # for i, p in enumerate(shape.points):
        #     if self.outOfPixmap(shape.rotatePoint(p,angle)):
        #         # print("out of pixmap")
        #         return
        if not self.rotateOutOfBound(angle):
            shape.rotate(angle)
            self.prevPoint = pos

    def getAngle(self, center, p1, p2):
        dx1 = p1.x() - center.x();
        dy1 = p1.y() - center.y();

        dx2 = p2.x() - center.x();
        dy2 = p2.y() - center.y();

        c = math.sqrt(dx1*dx1 + dy1*dy1) * math.sqrt(dx2*dx2 + dy2*dy2)
        if c == 0: return 0
        y = (dx1*dx2+dy1*dy2)/c
        if y>1: return 0
        angle = math.acos(y)

        if (dx1*dy2-dx2*dy1)>0:   
            return angle
        else:
            return -angle

    def boundedMoveShape(self, shape, pos):
        if shape.isRotated and self._can_out_of_bounding:
            c = shape.center
            dp = pos - self.prevPoint
            dc = c + dp
            if dc.x() < 0:
                dp -= QPointF(min(0,dc.x()), 0)
            if dc.y() < 0:                
                dp -= QPointF(0, min(0,dc.y()))                
            if dc.x() >= self.pixmap.width():
                dp += QPointF(min(0, self.pixmap.width() - 1  - dc.x()), 0)
            if dc.y() >= self.pixmap.height():
                dp += QPointF(0, min(0, self.pixmap.height() - 1 - dc.y()))

        else:            
            if self.out_of_pixmap(pos):
                return False  # No need to move
            o1 = pos + self.offsets[0]
            if self.out_of_pixmap(o1):
                pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
            o2 = pos + self.offsets[1]
            if self.out_of_pixmap(o2):
                pos += QPointF(min(0, self.pixmap.width() - 1 - o2.x()),
                               min(0, self.pixmap.height() - 1 - o2.y()))
            dp = pos - self.prevPoint
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        #self.calculateOffsets(self.selectedShape, pos)
        
        if dp:
            shape.move_by(dp)
            self.prevPoint = pos
            shape.close()
            return True
        return False


    def boundedMoveShape2(self, shape, pos):
        if self.out_of_pixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.out_of_pixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.out_of_pixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - o2.x()),
                           min(0, self.pixmap.height() - o2.y()))
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        #self.calculateOffsets(self.selectedShape, pos)
        dp = pos - self.prevPoint
        if dp:
            shape.move_by(dp)
            self.prevPoint = pos
            shape.close()
            return True
        return False

    def deSelectShape(self):
        if self.selected_shape:
            self.selected_shape.selected = False
            self.selected_shape = None
            self.setHiding(False)
            self.sig_shape_selection_changed.emit(False)
            self.update()

    def delete_selected(self):
        if self.selected_shape:
            shape = self.selected_shape
            self._shapes.remove(self.selected_shape)
            self.selected_shape = None
            self.update()
            return shape

    def get_longside_shortside_of_selected_box(self):
        if self.selected_shape:
            shape = self.selected_shape
            rect = shape.bounding_rect()
            long = rect.height() if rect.height() > rect.width() else rect.width()
            short = rect.height() if rect.height() < rect.width() else rect.width()
            self.update()
            return long, short

    def set_longside_shortside_of_selected_box(self, long, short):
        result_shape = None
        if self.selected_shape:
            shape = self.selected_shape
            is_need_confirm = shape.change_longside_shortside_of_shape_box(long, short)
            if is_need_confirm == False:
                return result_shape
            for p in shape.points:
                if self.out_of_pixmap(p):
                    self._shapes.remove(shape)
                    self.selected_shape = None
                    result_shape = shape
                    break
        self.update()
        return result_shape

    def set_longside_shortside_of_shape_box(self, shape, long, short):
        if shape:
            is_need_confirm = shape.change_longside_shortside_of_shape_box(long, short)
            if is_need_confirm == False:
                return None, False
            for p in shape.points:
                if self.out_of_pixmap(p):
                    return None, True
        return shape, True

    def copySelectedShape(self):
        if self.selected_shape:
            shape = self.selected_shape.copy()
            self.deSelectShape()
            self._shapes.append(shape)
            shape.selected = True
            self.selected_shape = shape
            self.boundedShiftShape(shape)
            return shape
        else:
            return None

    def append_shape(self, shape):
        self._shapes.append(shape)
        self.update()


    def boundedShiftShape(self, shape):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shape[0]
        offset = QPointF(2.0, 2.0)
        self.calculateOffsets(shape, point)
        self.prevPoint = point
        if not self.boundedMoveShape(shape, point - offset):
            self.boundedMoveShape(shape, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self._shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                if (shape.isRotated and not self.hideRotated) or (not shape.isRotated and not self.hideNormal):
                    shape.fill = shape.selected or shape == self.hShape
                    shape.paint(p)
                elif self.showCenter:
                    shape.fill = shape.selected or shape == self.hShape
                    shape.paintNormalCenter(p)

        if self._current_shape:
            self._current_shape.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        # Paint rect
        if self._current_shape is not None and len(self.line) == 2:
            leftTop = self.line[0]
            rightBottom = self.line[1]
            rectWidth = rightBottom.x() - leftTop.x()
            rectHeight = rightBottom.y() - leftTop.y()
            color = QColor(0, 220, 0)
            p.setPen(color)
            brush = QBrush(Qt.BDiagPattern)
            p.setBrush(brush)
            p.drawRect(leftTop.x(), leftTop.y(), rectWidth, rectHeight)
            
            #draw dialog line of rectangle
            p.setPen(self.lineColor)
            p.drawLine(leftTop.x(),rightBottom.y(),rightBottom.x(),leftTop.y())

        # self.setAutoFillBackground(True)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def out_of_pixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() < w and 0 <= p.y() < h)

    def finalise(self):
        assert self._current_shape
        self._current_shape.isRotated = self.canDrawRotatedRect
        # print(self.canDrawRotatedRect)
        self._current_shape.close()
        self._shapes.append(self._current_shape)
        self._current_shape = None
        self.setHiding(False)
        self.sig_new_shape_made.emit()
        self.update()

    def closeEnough(self, p1, p2):
        #d = distance(p1 - p2)
        #m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        return distance(p1 - p2) < self.epsilon

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width(), 0),
                  (size.width(), size.height()),
                  (0, size.height())]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)

    def intersectingEdges(self, x1y1, x2y2, points):
        """For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen."""
        x1, y1 = x1y1
        x2, y2 = x2y2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()
        if v_delta:
            self.sig_zoom_request.emit(v_delta, ev.pos())
        else:
            h_delta and self.sig_scroll_request.emit(h_delta, Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        
        if key == Qt.Key_Escape and self._current_shape:
            print('ESC press')
            self._current_shape = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()
        elif key == Qt.Key_Left:
            self._labeler.openPrevImg()
            # self.moveOnePixel('Left')
        elif key == Qt.Key_Right:
            self._labeler.openNextImg()
            # self.moveOnePixel('Right')
        elif key == Qt.Key_Up:
            self._labeler.openPrevImg()
            # self.moveOnePixel('Up')
        elif key == Qt.Key_Down:
            self._labeler.openNextImg()
            # self.moveOnePixel('Down')
        # elif key == Qt.Key_Z and self.selectedShape and\
        #      self.selectedShape.isRotated and not self.rotateOutOfBound(0.1):
        #     self.selectedShape.rotate(0.1)
        #     self.shapeMoved.emit()
        #     self.update()
        # elif key == Qt.Key_X and self.selectedShape and\
        #      self.selectedShape.isRotated and not self.rotateOutOfBound(0.01):
        #     self.selectedShape.rotate(0.01)
        #     self.shapeMoved.emit()
        #     self.update()
        # elif key == Qt.Key_C and self.selectedShape and\
        #      self.selectedShape.isRotated and not self.rotateOutOfBound(-0.01):
        #     self.selectedShape.rotate(-0.01)
        #     self.shapeMoved.emit()
        #     self.update()
        # elif key == Qt.Key_V and self.selectedShape and\
        #      self.selectedShape.isRotated and not self.rotateOutOfBound(-0.1):
        #     self.selectedShape.rotate(-0.1)
        #     self.shapeMoved.emit()
        #     self.update()
        elif key == Qt.Key_R:
            self.hideRotated = not self.hideRotated
            self.hideRRect.emit(self.hideRotated)
            self.update()
        elif key == Qt.Key_N:
            self.hideNormal = not self.hideNormal
            self.hideNRect.emit(self.hideNormal)
            self.update()
        elif key == Qt.Key_O:
            self._can_out_of_bounding = not self._can_out_of_bounding
        elif key == Qt.Key_B:
            self.showCenter = not self.showCenter
            self.update()


    def rotateOutOfBound(self, angle):
        if self._can_out_of_bounding:
            return False
        for i, p in enumerate(self.selected_shape.points):
            if self.out_of_pixmap(self.selected_shape.rotatePoint(p, angle)):
                return True
        return False

    def moveOnePixel(self, direction):
        # print(self.selectedShape.points)
        if direction == 'Left' and not self.moveOutOfBound(QPointF(-1.0, 0)):
            # print("move Left one pixel")
            self.selected_shape.points[0] += QPointF(-1.0, 0)
            self.selected_shape.points[1] += QPointF(-1.0, 0)
            self.selected_shape.points[2] += QPointF(-1.0, 0)
            self.selected_shape.points[3] += QPointF(-1.0, 0)
            self.selected_shape.center += QPointF(-1.0, 0)
        elif direction == 'Right' and not self.moveOutOfBound(QPointF(1.0, 0)):
            # print("move Right one pixel")
            self.selected_shape.points[0] += QPointF(1.0, 0)
            self.selected_shape.points[1] += QPointF(1.0, 0)
            self.selected_shape.points[2] += QPointF(1.0, 0)
            self.selected_shape.points[3] += QPointF(1.0, 0)
            self.selected_shape.center += QPointF(1.0, 0)
        elif direction == 'Up' and not self.moveOutOfBound(QPointF(0, -1.0)):
            # print("move Up one pixel")
            self.selected_shape.points[0] += QPointF(0, -1.0)
            self.selected_shape.points[1] += QPointF(0, -1.0)
            self.selected_shape.points[2] += QPointF(0, -1.0)
            self.selected_shape.points[3] += QPointF(0, -1.0)
            self.selected_shape.center += QPointF(0, -1.0)
        elif direction == 'Down' and not self.moveOutOfBound(QPointF(0, 1.0)):
            # print("move Down one pixel")
            self.selected_shape.points[0] += QPointF(0, 1.0)
            self.selected_shape.points[1] += QPointF(0, 1.0)
            self.selected_shape.points[2] += QPointF(0, 1.0)
            self.selected_shape.points[3] += QPointF(0, 1.0)
            self.selected_shape.center += QPointF(0, 1.0)
        self.shapeMoved.emit()
        self.update()

    def moveOutOfBound(self, step):
        points = [p1 + p2 for p1, p2 in zip(self.selected_shape.points, [step] * 4)]
        return True in map(self.out_of_pixmap, points)

    def setLastLabel(self, text):
        assert text
        self._shapes[-1].set_label(text)
        return self._shapes[-1]

    def undoLastLine(self):
        assert self._shapes
        self._current_shape = self._shapes.pop()
        self._current_shape.setOpen()
        self.line.points = [self._current_shape[-1], self._current_shape[0]]
        self.drawingPolygon.emit(True)

    def resetAllLines(self):
        assert self._shapes
        self._current_shape = self._shapes.pop()
        self._current_shape.setOpen()
        self.line.points = [self._current_shape[-1], self._current_shape[0]]
        self.drawingPolygon.emit(True)
        self._current_shape = None
        self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self._shapes = []
        self.update()



    #This is called when a file is loaded.
    def loadShapes(self, shapes):
        self._shapes = list(shapes)
        self._current_shape = None
        self.update()


    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        if self._cursor != cursor:
            self.restoreCursor()
            self._cursor = cursor
            QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        self._cursor = CURSOR_DEFAULT
        QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.update()


    def get_shape(self):
        return deepcopy(self._shapes)


