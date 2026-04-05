from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget

from aligner_gui.shared import gui_util


class GraphWidget(QWidget):

    styles = ['r', 'g', 'y', 'm', 'k', 'c']

    def __init__(self, **kargs):
        super().__init__()
        self.data = {}

        self.hideLegend = kargs['hide_legend'] if 'hide_legend' in kargs else None
        self.hideBottom = kargs['hide_bottom'] if 'hide_bottom' in kargs else None
        self.limitRange = kargs['limit_range'] if 'limit_range' in kargs else None
        self.plotWidget = None
        self._pg = None

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")

        self.cur_style = 0
        self.reset()

    def _get_pg(self):
        if self._pg is None:
            import pyqtgraph as pg

            self._pg = pg
        return self._pg

    def initPlotWidget(self):
        pg = self._get_pg()
        if self.plotWidget is not None:
            self.plotWidget.close()

        gui_util.reset_layout(self.verticalLayout)

        self.plotWidget = pg.PlotWidget()
        self.verticalLayout.addWidget(self.plotWidget)

        pg.setConfigOptions(antialias=True)
        self.plotWidget.setBackground((0, 0, 0, 0))
        self.plotWidget.setMenuEnabled(False)
        self.plotWidget.setMouseEnabled(x=False, y=False)
        self.plotWidget.plotItem.showGrid(x=True, y=True, alpha=0.18)
        self.plotWidget.plotItem.getAxis('left').setPen(pg.mkPen('#a8b3bf'))
        self.plotWidget.plotItem.getAxis('left').setTextPen(pg.mkPen('#d7dee6'))
        self.plotWidget.plotItem.getAxis('bottom').setPen(pg.mkPen('#a8b3bf'))
        self.plotWidget.plotItem.getAxis('bottom').setTextPen(pg.mkPen('#d7dee6'))
        self.plotWidget.plotItem.getAxis('bottom').setLabel('Step')

        if self.hideLegend is None:
            legend = self.plotWidget.plotItem.addLegend()
            legend.setBrush(pg.mkBrush(15, 15, 15, 180))
            legend.setPen(pg.mkPen(90, 90, 90, 180))

        if self.hideBottom is not None:
            self.plotWidget.plotItem.hideAxis('bottom')

        if self.limitRange is not None:
            self.plotWidget.setYRange(min=self.limitRange[0], max=self.limitRange[1])

    def _ensure_plot_widget(self):
        if self.plotWidget is None:
            self.initPlotWidget()

    def setName(self, name):
        self._ensure_plot_widget()
        self.plotWidget.plotItem.setLabel('left', text=name)

    def setBottomName(self, name):
        self._ensure_plot_widget()
        self.plotWidget.plotItem.getAxis('bottom').setLabel(name)

    def reset(self):
        self.data.clear()
        self.cur_style = 0
        if self.plotWidget is not None:
            self.initPlotWidget()

    @staticmethod
    def _to_xy(data):
        x = list(range(1, len(data) + 1))
        return x, data

    def _update_epoch_axis(self, data_len: int):
        self._ensure_plot_widget()
        if data_len <= 0:
            self.plotWidget.plotItem.getAxis('bottom').setTicks(None)
            return

        axis = self.plotWidget.plotItem.getAxis('bottom')
        max_x = max(data_len, 2)
        self.plotWidget.setXRange(1, max_x, padding=0.02)

        if data_len <= 10:
            step = 1
        elif data_len <= 20:
            step = 2
        elif data_len <= 50:
            step = 5
        elif data_len <= 100:
            step = 10
        else:
            step = max(10, int(round(data_len / 10.0)))

        ticks = [(float(x), str(x)) for x in range(1, data_len + 1, step)]
        if ticks[-1][0] != float(data_len):
            ticks.append((float(data_len), str(data_len)))
        axis.setTicks([ticks])

    def setData(self, data: dict):
        self._ensure_plot_widget()
        # self.reset()
        max_len = 0
        for key in data.keys():
            if key not in self.data.keys():
                pen = self.styles[self.cur_style]
                self.cur_style = (self.cur_style + 1) % len(self.styles)
                self.data[key] = self.plotWidget.plotItem.plot(pen=pen, name=key)

            line = self.data[key]
            x, y = self._to_xy(data[key])
            max_len = max(max_len, len(x))
            line.setData(x=x, y=y)
        self._update_epoch_axis(max_len)

    def setLine(self, name, data, color, width):
        self._ensure_plot_widget()
        pg = self._get_pg()
        if name not in self.data.keys():
            pen = pg.mkPen(color=color, width=width)
            self.data[name] = self.plotWidget.plotItem.plot(
                pen=pen,
                name=name,
                symbol='o',
                symbolSize=5,
                symbolBrush=color,
                symbolPen=pen,
            )
        x, y = self._to_xy(data)
        if len(data) > 40:
            self.data[name].setSymbol(None)
        else:
            self.data[name].setSymbol('o')
            self.data[name].setSymbolSize(5)
        self.data[name].setData(x=x, y=y)
        self._update_epoch_axis(len(x))

"""
    def setMarker(self, name, markers, color, width):
        if name not in self.data.keys():
            pen = pg.mkPen(color=color, width=width)
            self.data[name] = self.plotWidget.plotItem.plot(pen=pen, name=name)
        self.data[name].setData(markers, )
        """


