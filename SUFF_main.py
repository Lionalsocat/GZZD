from PyQt5.Qt import *
from UI.SUFF_ui import Ui_MainWindow
from TB_main import TB_MainWind
import sys
import os
from Code.suff_deal import get_acc_loss

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolBar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import subprocess


# 初始化的画板，设置字体，SizePolicy等
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=10, plot_type=''):
        # 显示中文和负号
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        PlotCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.acc_axes = self.fig.add_subplot(121)
        self.acc_axes.plot([])
        self.acc_axes.set_title(plot_type+'_acc')

        self.loss_axes = self.fig.add_subplot(122)
        self.loss_axes.plot([])
        self.loss_axes.set_title(plot_type+'_loss')


class SUFF_Wind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        # self.setFixedSize(800, 800)

        self.semi_sup_fig = PlotCanvas(width=5, height=3, dpi=70, plot_type='semi_sup')
        self.semi_sup_fig_ntb = NavigationToolBar(self.semi_sup_fig, self)
        self.sup_fig = PlotCanvas(width=5, height=3, dpi=70, plot_type='sup')
        self.sup_fig_ntb = NavigationToolBar(self.sup_fig, self)
        self.unsup_fig = PlotCanvas(width=5, height=3, dpi=70, plot_type='unsup')
        self.unsup_fig_ntb = NavigationToolBar(self.unsup_fig, self)

        self.semi_sup_Layout.addWidget(self.semi_sup_fig)
        self.semi_sup_Layout.addWidget(self.semi_sup_fig_ntb)
        self.sup_Layout.addWidget(self.sup_fig)
        self.sup_Layout.addWidget(self.sup_fig_ntb)
        self.unsup_Layout.addWidget(self.unsup_fig)
        self.unsup_Layout.addWidget(self.unsup_fig_ntb)

        self.sup_logs_path = ''
        self.unsup_logs_path = ''
        self.semi_sup_logs_path = ''

        self.sup_btn.clicked.connect(self.show_sup_pic)
        self.unsup_btn.clicked.connect(self.show_unsup_pic)
        self.semi_sup_btn.clicked.connect(self.show_semi_sup_pic)

        self.choice_sup.clicked.connect(self.sup_load_path)
        self.choice_unsup.clicked.connect(self.unsup_load_path)
        self.choice_semi_sup.clicked.connect(self.semi_sup_load_path)

        self.sup_tb.clicked.connect(self.open_tb_sup)
        self.sup_tb_close.clicked.connect(self.close_tb_sup)
        self.semisup_tb.clicked.connect(self.open_tb_semi_sup)
        self.semisup_tb_close.clicked.connect(self.close_tb_semi_sup)
        self.unsup_tb.clicked.connect(self.open_tb_unsup)
        self.unsup_tb_close.clicked.connect(self.close_tb_unsup)

    def sup_load_path(self):
        self.sup_logs_path = QFileDialog.getExistingDirectory(self, '选择要显示结果的监督exp', './result/Supervised')

    def unsup_load_path(self):
        self.unsup_logs_path = QFileDialog.getExistingDirectory(self, '选择要显示结果的无监督exp', './result/Unsupervised')

    def semi_sup_load_path(self):
        self.semi_sup_logs_path = QFileDialog.getExistingDirectory(self, '选择要显示结果的半监督exp', './result/Semi_supervised')

    def show_sup_pic(self):
        if self.sup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.sup_logs_path, 'runs.txt')
        if not os.path.isfile(log_path):
            QMessageBox.warning(self, 'warning', '未进行监督学习训练， 无结果！')
            return
        sup_train_acc, sup_train_loss, sup_val_acc, sup_val_loss = get_acc_loss(log_path)

        self.sup_fig.acc_axes.cla()
        self.sup_fig.acc_axes.plot(sup_train_acc, label='Training Acc')
        self.sup_fig.acc_axes.plot(sup_val_acc, label='Validation Acc')
        self.sup_fig.acc_axes.set_title('sup_acc')
        self.sup_fig.acc_axes.set_xlabel('epochs')
        self.sup_fig.acc_axes.set_ylabel('acc')
        self.sup_fig.acc_axes.legend()

        self.sup_fig.loss_axes.cla()
        self.sup_fig.loss_axes.plot(sup_train_loss, label='Training Loss')
        self.sup_fig.loss_axes.plot(sup_val_loss, label='Validation Loss')
        self.sup_fig.loss_axes.set_title('sup_loss')
        self.sup_fig.loss_axes.set_xlabel('epochs')
        self.sup_fig.loss_axes.set_ylabel('loss')
        self.sup_fig.loss_axes.legend()

        self.sup_fig.draw()

    def show_unsup_pic(self):
        if self.unsup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.unsup_logs_path, 'runs.txt')
        if not os.path.isfile(log_path):
            QMessageBox.warning(self, 'warning', '未进行无监督学习训练， 无结果！')
            return
        unsup_train_acc, unsup_train_loss, unsup_val_acc, unsup_val_loss = get_acc_loss(log_path)

        self.unsup_fig.acc_axes.cla()
        self.unsup_fig.acc_axes.plot(unsup_train_acc, label='Training Acc')
        self.unsup_fig.acc_axes.plot(unsup_val_acc, label='Validation Acc')
        self.unsup_fig.acc_axes.set_title('unsup_acc')
        self.unsup_fig.acc_axes.set_xlabel('epochs')
        self.unsup_fig.acc_axes.set_ylabel('acc')
        self.unsup_fig.acc_axes.legend()

        self.unsup_fig.loss_axes.cla()
        self.unsup_fig.loss_axes.plot(unsup_train_loss, label='Training Loss')
        self.unsup_fig.loss_axes.plot(unsup_val_loss, label='Validation Loss')
        self.unsup_fig.loss_axes.set_title('unsup_loss')
        self.unsup_fig.loss_axes.set_xlabel('epochs')
        self.unsup_fig.loss_axes.set_ylabel('loss')
        self.unsup_fig.loss_axes.legend()

        self.unsup_fig.draw()

    def show_semi_sup_pic(self):
        if self.semi_sup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.semi_sup_logs_path, 'runs.txt')
        if not os.path.isfile(log_path):
            QMessageBox.warning(self, 'warning', '未进行半监督学习训练， 无结果！')
            return
        semi_sup_train_acc, semi_sup_train_loss, semi_sup_val_acc, semi_sup_val_loss = get_acc_loss(log_path)

        self.semi_sup_fig.acc_axes.cla()
        self.semi_sup_fig.acc_axes.plot(semi_sup_train_acc, label='Training Acc')
        self.semi_sup_fig.acc_axes.plot(semi_sup_val_acc, label='Validation Acc')
        self.semi_sup_fig.acc_axes.set_title('semi_sup_acc')
        self.semi_sup_fig.acc_axes.set_xlabel('epochs')
        self.semi_sup_fig.acc_axes.set_ylabel('acc')
        self.semi_sup_fig.acc_axes.legend()

        self.semi_sup_fig.loss_axes.cla()
        self.semi_sup_fig.loss_axes.plot(semi_sup_train_loss, label='Training Loss')
        self.semi_sup_fig.loss_axes.plot(semi_sup_val_loss, label='Validation Loss')
        self.semi_sup_fig.loss_axes.set_title('semi_sup_loss')
        self.semi_sup_fig.loss_axes.set_xlabel('epochs')
        self.semi_sup_fig.loss_axes.set_ylabel('loss')
        self.semi_sup_fig.loss_axes.legend()

        self.semi_sup_fig.draw()

    def open_tb_sup(self):

        if self.sup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.sup_logs_path, 'logs')
        if not os.path.exists(log_path):
            QMessageBox.warning(self, 'warning', '未进行监督学习训练， 无结果！')
            return
        cmd_command = 'tensorboard --logdir ' + log_path + ' --host=127.0.0.1 --port=6006'
        self.sup_pipe = subprocess.Popen(cmd_command, shell=False)
        web_site = r'http://127.0.0.1:6006/'
        self.tb_wind_sup = TB_MainWind(web_site)
        self.tb_wind_sup.tb_web.load(QUrl(web_site))
        self.tb_wind_sup.show()

        self.sup_tb.setEnabled(False)
        self.sup_tb_close.setEnabled(True)
        self.choice_sup.setEnabled(False)

    def open_tb_semi_sup(self):

        if self.semi_sup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.semi_sup_logs_path, 'logs')
        if not os.path.exists(log_path):
            QMessageBox.warning(self, 'warning', '未进行半监督学习训练， 无结果！')
            return
        cmd_command = 'tensorboard --logdir ' + log_path + ' --host=127.0.0.1 --port=6007'
        self.semi_sup_pipe = subprocess.Popen(cmd_command, shell=False)
        web_site = r'http://127.0.0.1:6007/'
        self.tb_wind_semi_sup = TB_MainWind(web_site)
        self.tb_wind_semi_sup.tb_web.load(QUrl(web_site))
        self.tb_wind_semi_sup.show()

        self.semisup_tb.setEnabled(False)
        self.semisup_tb_close.setEnabled(True)
        self.choice_semi_sup.setEnabled(False)

    def open_tb_unsup(self):

        if self.unsup_logs_path == '':
            QMessageBox.warning(self, 'warning', '未选择exp文件夹， 无结果！')
            return
        log_path = os.path.join(self.unsup_logs_path, 'logs')
        if not os.path.exists(log_path):
            QMessageBox.warning(self, 'warning', '未进行无监督学习训练， 无结果！')
            return
        cmd_command = 'tensorboard --logdir ' + log_path + ' --host=127.0.0.1 --port=6008'
        self.unsup_pipe = subprocess.Popen(cmd_command, shell=False)
        web_site = r'http://127.0.0.1:6008/'
        self.tb_wind_unsup = TB_MainWind(web_site)
        self.tb_wind_unsup.tb_web.load(QUrl(web_site))
        self.tb_wind_unsup.show()

        self.unsup_tb.setEnabled(False)
        self.unsup_tb_close.setEnabled(True)
        self.choice_unsup.setEnabled(False)

    def close_tb_sup(self):
        self.sup_pipe.terminate()
        self.sup_tb.setEnabled(True)
        self.sup_tb_close.setEnabled(False)
        self.choice_sup.setEnabled(True)

    def close_tb_semi_sup(self):
        self.semi_sup_pipe.terminate()
        self.semisup_tb.setEnabled(True)
        self.semisup_tb_close.setEnabled(False)
        self.choice_semi_sup.setEnabled(True)

    def close_tb_unsup(self):
        self.unsup_pipe.terminate()
        self.unsup_tb.setEnabled(True)
        self.unsup_tb_close.setEnabled(False)
        self.choice_unsup.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wind = SUFF_Wind()
    wind.show()
    sys.exit(app.exec_())
