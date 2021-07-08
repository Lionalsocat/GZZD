from PyQt5.Qt import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
import sys
import os
from UI.UNSUP_ui import Ui_MainWindow
from TB_main import TB_MainWind
from Code.Unsupervised.train import train, draw_acc_loss, draw_acc_loss_ani
from Code.Unsupervised.predict import predict
import subprocess
from Code.suff_deal import get_acc_loss

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolBar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Result_Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=10):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(Result_Canvas, self).__init__(self.fig)
        self.setParent(parent)
        Result_Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.acc_axes = self.fig.add_subplot(121)
        self.acc_axes.plot([])
        self.acc_axes.set_title('acc')

        self.loss_axes = self.fig.add_subplot(122)
        self.loss_axes.plot([])
        self.loss_axes.set_title('loss')


class UNSUP_Wind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.data_file_path = ''
        self.data_test_path = ''
        self.model_path = ''

        self.code_root = './Code/Unsupervised'
        self.model_save_path = './result/Unsupervised'

        self.result_plot = Result_Canvas(width=5, height=3, dpi=70)
        self.result_plot_ntb = NavigationToolBar(self.result_plot, self)
        self.intime_layout.addWidget(self.result_plot)
        self.intime_layout.addWidget(self.result_plot_ntb)

        self.train_progressBar.setValue(0)
        self.train_progressBar.setHidden(True)

        self.load_data_btn.clicked.connect(self.load_data)
        self.load_model_btn.clicked.connect(self.load_model)
        self.train_btn.clicked.connect(self.train)
        self.load_test_btn.clicked.connect(self.load_test)
        self.predict_btn.clicked.connect(self.predict)
        self.show_tb_btn.clicked.connect(self.load_tb)

    def load_data(self):
        self.data_file_path = QFileDialog.getExistingDirectory(self, '选择数据所在文件夹', './')

    def load_test(self):
        self.data_test_path, _ = QFileDialog.getOpenFileName(self, "加载数据", './', "MAT Files(*.mat)")

    def load_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, "加载权重", self.model_save_path, "h5 Files(*.h5)")

    def train(self):

        if self.ep_whole_edit.text() == '' or self.ep_layer_edit.text() == '' or self.bs_edit.text() == '':
            QMessageBox.warning(self, 'warning', '请输入对应数值！')
            return
        if self.data_file_path == '' :
            QMessageBox.warning(self, 'warning', '未选择数据根目录')
            return

        if not os.path.isfile(os.path.join(self.data_file_path, 'data_train.npy')):
            QMessageBox.warning(self, 'warning', 'data_train.npy 不存在!')
            return
        if not os.path.isfile(os.path.join(self.data_file_path, 'data_val.npy')):
            QMessageBox.warning(self, 'warning', 'data_val.npy 不存在!')
            return

        classes_num = -1
        ori_dim = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'num_classes':
                        classes_num = int(item[1])
                    if item[0] == 'data_length':
                       ori_dim = int(item[1])
            if classes_num == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少num_classes数值')
                return
            if ori_dim == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少data_length数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        if self.save_last_only.isChecked():
            save_last = True
        else:
            save_last = False
        epcohs_layer = int(self.ep_layer_edit.text())
        epochs_whole = int(self.ep_whole_edit.text())
        self.epochs = epochs_whole
        bs = int(self.bs_edit.text())

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        exp_files = [temp for temp in os.listdir(self.model_save_path) if
                     os.path.isdir(os.path.join(self.model_save_path, temp)) and temp[:3] == 'exp']
        if len(exp_files) == 0:
            save_dir = os.path.join(self.model_save_path, 'exp1')
        else:
            exp_n = [int(item[3:]) for item in exp_files]
            exp_n.sort()
            now_n = exp_n[-1] + 1
            save_dir = os.path.join(self.model_save_path, f'exp{now_n}')

        # 保存训练参数
        with open(os.path.join(self.model_save_path, 'train_info.txt'), 'a') as f:
            f.write(f'model_save_path:{save_dir};  epochs_layer:{epcohs_layer};  epochs_whole:{epochs_whole};'
                    f'  classes_num:{classes_num};  original_dim:{ori_dim};  batch_size:{bs}\n')

        # 初始化train线程
        self.train_thread = Train_thread(data_path=self.data_file_path, classes_num=classes_num, ori_dim=ori_dim,
                                         epochs_whole=epochs_whole, epochs_layer=epcohs_layer, batch_size=bs,
                                         model_save_path=save_dir, save_last=save_last)
        self.train_thread.sig.connect(self.draw_result)

        # 初始化画图线程
        self.draw_thread = Draw_in_time_thread(runs_root=os.path.join(save_dir, 'runs.txt'))
        self.draw_thread.sig.connect(self.draw_in_time)

        # 打开Tensorboard的CMD
        log_dir = os.path.join(save_dir, 'logs')
        cmd_command = 'tensorboard --logdir ' + log_dir + ' --host=127.0.0.1 --port=6006'
        self.pipe = subprocess.Popen(cmd_command, shell=False)
        # 训练开始
        self.train_thread.start()
        self.train_progressBar.setValue(0)
        self.train_progressBar.setHidden(False)
        self.show_tb_btn.setEnabled(True)
        self.draw_thread.start()

        self.train_btn.setEnabled(False)

    def draw_in_time(self, result_tuple):
        acc, loss, val_acc, val_loss = result_tuple[0]
        now_epoch = int(result_tuple[1])
        total_epoch = int(self.epochs)
        self.train_progressBar.setValue((now_epoch/total_epoch)*100)

        self.result_plot.acc_axes.cla()
        self.result_plot.acc_axes.plot(acc, label='Training Acc')
        self.result_plot.acc_axes.plot(val_acc, label='Validation Acc')
        self.result_plot.acc_axes.set_title('unsup_acc')
        self.result_plot.acc_axes.set_xlabel('epochs')
        self.result_plot.acc_axes.set_ylabel('acc')
        self.result_plot.acc_axes.legend()

        self.result_plot.loss_axes.cla()
        self.result_plot.loss_axes.plot(loss, label='Training Loss')
        self.result_plot.loss_axes.plot(val_loss, label='Validation Loss')
        self.result_plot.loss_axes.set_title('unsup_loss')
        self.result_plot.loss_axes.set_xlabel('epochs')
        self.result_plot.loss_axes.set_ylabel('loss')
        self.result_plot.loss_axes.legend()

        self.result_plot.draw()

    def draw_result(self, dict, save_path):

        pic_save_path = os.path.join(save_path, 'UNSUP_result.jpg')
        gif_save_path = os.path.join(save_path, 'UNSUP_result.gif')
        draw_acc_loss(dict, pic_save_path)
        draw_acc_loss_ani(dict, gif_save_path)
        QMessageBox.information(self, 'INFO', '训练完成!')
        # 退出线程
        self.train_thread.quit()
        self.draw_thread.quit()
        self.train_progressBar.setHidden(True)
        self.show_tb_btn.setEnabled(False)
        # 结束tensorboard
        self.pipe.terminate()

        self.train_btn.setEnabled(True)

    def predict(self):

        if self.data_test_path == '':
            QMessageBox.warning(self, 'warning', '未加载数据！')
            return
        if self.model_path == '':
            QMessageBox.warning(self, 'warning', '未加载模型！')
            return

        ori_dim = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'data_length':
                        ori_dim = int(item[1])
            if ori_dim == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少data_length数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        clas, clas_pro = predict(self.model_path, self.data_test_path, ori_dim)
        print(clas, clas_pro)

        if os.path.isfile('./data/classes.txt'):
            with open('./data/classes.txt', 'r') as f:
                classes_name = [tmp.strip() for tmp in f.readlines()]
            self.clas_edit.setText(str(classes_name[clas]))
        else:
            self.clas_edit.setText(str(clas))
        self.clas_pro_edit.setText(str(clas_pro))

    def load_tb(self):
        web_site = r'http://127.0.0.1:6006/'
        self.tb_wind = TB_MainWind(web_site)
        self.tb_wind.tb_web.load(QUrl(web_site))
        self.tb_wind.show()


class Train_thread(QThread):

    sig = pyqtSignal(dict, str)

    def __init__(self, data_path, classes_num, ori_dim, epochs_layer, epochs_whole,
                 batch_size, model_save_path, save_last):
        super(Train_thread, self).__init__()
        self.data_path = data_path
        self.classes_num = classes_num
        self.ori_dim = ori_dim
        self.epochs_layer = epochs_layer
        self.epochs_whole = epochs_whole
        self.bs = batch_size
        self.model_save_path = model_save_path
        self.save_last = save_last

    def run(self):
        result_dict = train(self.data_path, self.classes_num, ori_dim=self.ori_dim, epochs_layer=self.epochs_layer,
                            epochs_whole=self.epochs_whole, batch_size=self.bs, model_save_path=self.model_save_path,
                            save_last=self.save_last)
        self.sig.emit(result_dict, self.model_save_path)


class Draw_in_time_thread(QThread):
    sig = pyqtSignal(tuple)

    def __init__(self, runs_root):
        super(Draw_in_time_thread, self).__init__()
        self.runs_root = runs_root

    def run(self):
        record_len = 0
        while True:
            if not os.path.exists(self.runs_root):
                continue
            record = get_acc_loss(self.runs_root)
            if len(record[0]) > record_len:
                record_len = len(record[0])
                self.sig.emit((record, record_len))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_gui = UNSUP_Wind()
    video_gui.show()
    sys.exit(app.exec_())
