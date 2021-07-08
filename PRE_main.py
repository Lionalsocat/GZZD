from PyQt5.Qt import *
from UI.PRE_ui import Ui_MainWindow
import sys
import scipy.io as scio
from Code.pre_deal import deal_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolBar


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=10):
        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        PlotCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.init_fig()

    def init_fig(self):
        pass


class SPCanvas(PlotCanvas):
    def __init__(self, *args, **kwargs):
        super(SPCanvas, self).__init__(*args, **kwargs)

    def init_fig(self):
        self.sy_axes = self.fig.add_subplot(211)
        self.sy_axes.plot([])
        self.sy_axes.set_title('时域信号')

        self.py_axes = self.fig.add_subplot(212)
        self.py_axes.plot([])
        self.py_axes.set_title('频域信号')


class DISCanvas(PlotCanvas):
    def __init__(self, *args, **kwargs):
        super(DISCanvas, self).__init__(*args, **kwargs)

    def init_fig(self):
        self.dis_axes = self.fig.add_subplot(111)
        self.dis_axes.bar([], [])
        self.dis_axes.set_title('数据分布')


class PRE_wind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.data_folder = ''
        self.data_one = ''

        self.SP_fig = SPCanvas(width=5, height=3, dpi=70)
        self.SP_ntb = NavigationToolBar(self.SP_fig, self)
        self.DIS_fig = DISCanvas(width=5, height=5, dpi=70)

        self.SP_Layout.addWidget(self.SP_fig)
        self.SP_Layout.addWidget(self.SP_ntb)
        self.Dis_Layout.addWidget(self.DIS_fig)

        self.select_floder_btn.clicked.connect(self.select_folder)
        self.deal_btn.clicked.connect(self.deal)
        self.load_btn.clicked.connect(self.load_one)
        self.draw_btn.clicked.connect(self.draw)

    def select_folder(self):
        self.data_folder = QFileDialog.getExistingDirectory(self, '选择数据根目录', './')

    def deal(self):

        data_long = self.data_long_edit.text()
        overlap = self.overlap_edit.text()
        val_pre = self.val_pre_edit.text()
        unlabel_pre = self.unlabel_edit.text()
        plot_dis = self.plot_check.isChecked()

        if self.data_folder == '' :
            QMessageBox.warning(self, 'warning', '未选择数据根目录')
            return
        if data_long == '' or overlap == '' or val_pre == '' or unlabel_pre == '':
            QMessageBox.warning(self, 'warning', '请输入对应数值')
            return
        if int(overlap) < 0 or int(overlap) > 100 or int(val_pre) < 0 or int(val_pre) > 100 \
                or int(unlabel_pre) < 0 or int(unlabel_pre) > 100:
            QMessageBox.warning(self, 'warning', '重叠率/验证集比例/无标签比例 范围: 0 - 100')
            return

        if self.shiyu_btn.isChecked():
            is_shi = True
        if self.piny_btn.isChecked():
            is_shi = False

        classes, data_dis = deal_data(self.data_folder, int(data_long), int(overlap), int(val_pre),
                                      int(unlabel_pre), plot_dis, is_shi)
        if plot_dis:
            self.DIS_fig.dis_axes.cla()

            self.DIS_fig.dis_axes.bar(range(len(classes)), data_dis, align='center')
            self.DIS_fig.dis_axes.set_xticks(range(len(classes)))
            self.DIS_fig.dis_axes.set_xticklabels(classes, rotation=0, fontsize=12)
            for i, v in enumerate(data_dis):
                self.DIS_fig.dis_axes.text(x=i, y=v+5, s=str(v), ha='center', fontsize=8)
            self.DIS_fig.dis_axes.set_xlabel('class', fontsize=10)
            self.DIS_fig.dis_axes.set_ylabel('number of data', fontsize=10)
            self.DIS_fig.dis_axes.set_title('class distribution', fontsize=12)

            self.DIS_fig.draw()

    def load_one(self):
        self.data_one, _ = QFileDialog.getOpenFileName(self, '选择数据', self.data_folder, 'mat文件(*.mat)')

    def draw(self):
        if self.data_one == '':
            QMessageBox.warning(self, 'warning', '未指定文件路径')
            return
        else:
            if self.shiyu_btn.isChecked():
                is_shi = True
            if self.piny_btn.isChecked():
                is_shi = False

            data_name = self.data_one.split('/')[-1]
            self.data_name_label.setText(data_name)

            data = scio.loadmat(self.data_one)
            x = data['x'][:, 0]
            if not is_shi:
                x = np.fft.ifft(x)
            N = len(x)

            self.SP_fig.sy_axes.cla()
            self.SP_fig.sy_axes.plot(x)
            self.SP_fig.sy_axes.set_title('时域图')

            self.SP_fig.py_axes.cla()
            t = np.linspace(-N / 2, N / 2, N)
            xf = np.fft.fft(x)
            xf_abs = np.fft.fftshift(abs(xf)/N)
            self.SP_fig.py_axes.plot(t, xf_abs)
            self.SP_fig.py_axes.set_title('频域图')

            self.SP_fig.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wind = PRE_wind()
    wind.show()
    sys.exit(app.exec_())
