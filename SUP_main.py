from PyQt5.Qt import *
import sys
import os
import numpy as np
from UI.SUP_ui import Ui_MainWindow
from TB_main import TB_MainWind
from Code.Supervised.train import train
from Code.Supervised.utils import signal2pic, draw_acc_loss, draw_acc_loss_ani
from Code.Supervised.predict import predict
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


class SUP_Wind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.code_root = './Code/Supervised/'
        self.model_save_path = './result/Supervised'

        self.data_path = ''
        self.model_path = ''
        self.test_data = ''

        self.result_plot = Result_Canvas(width=5, height=3, dpi=70)
        self.result_plot_ntb = NavigationToolBar(self.result_plot, self)
        self.intime_layout.addWidget(self.result_plot)
        self.intime_layout.addWidget(self.result_plot_ntb)

        self.singal2pic.stateChanged.connect(self.s2pchanged)

        self.train_progressBar.setValue(0)
        self.train_progressBar.setHidden(True)

        self.load_data_btn.clicked.connect(self.load_data)
        self.trai_button.clicked.connect(self.train)
        self.predict_b.clicked.connect(self.predict)
        self.load_test_b.clicked.connect(self.load_test)
        self.model_b.clicked.connect(self.load_model)
        self.show_tb_btn.clicked.connect(self.load_tb)
        self.convert2pic.clicked.connect(self.s2p_start)

    def s2pchanged(self):
        self.convert2pic.setEnabled(not self.convert2pic.isEnabled())

    def load_data(self):
        self.data_path = QFileDialog.getExistingDirectory(self, '选择数据所在文件夹', './')

    def load_test(self):
        self.test_data, _ = QFileDialog.getOpenFileName(self, '选择测试样本', './', "MAT Files(*.mat)")

    def load_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, '选择模型文件', self.model_save_path, "H5 Files(*.h5)")

    def s2p_start(self):

        if self.data_path == '':
            QMessageBox.warning(self, 'warning', '未生成pic数据集， 则需要手动指定数据文件夹！')
            return

        classes_num = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'num_classes':
                        classes_num = int(item[1])
            if classes_num == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少num_classes数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        train_data_path = os.path.join(self.data_path, 'data_train.npy')
        train_data_temp = np.load(train_data_path)
        self.train_n = len(train_data_temp)
        val_data_path = os.path.join(self.data_path, 'data_val.npy')
        val_data_temp = np.load(val_data_path)
        self.val_n = len(val_data_temp)

        self.s2p_thread = S2P_thread(train_path=train_data_path, val_path=val_data_path,
                                     code_path=self.code_root, classes_num=classes_num)
        self.s2p_thread.sig.connect(self.s2p_finish)

        if os.path.isfile(os.path.join(self.code_root, 'data/data_list.txt')):
            os.remove(os.path.join(self.code_root, 'data/data_list.txt'))

        self.convert_thread = Convert_in_time_thread(code_root=self.code_root)
        self.convert_thread.sig.connect(self.update_deal_data)

        self.s2p_thread.start()
        self.convert_thread.start()
        self.trai_button.setEnabled(False)
        self.convert2pic.setEnabled(False)
        self.singal2pic.setEnabled(False)
        self.train_progressBar.setValue(0)
        self.train_progressBar.setHidden(False)

    def update_deal_data(self, dealed_len):
        self.train_progressBar.setValue((dealed_len/(self.train_n+self.val_n))*100)

    def s2p_finish(self):
        QMessageBox.information(self, 'INFO', '转换完成!')
        self.trai_button.setEnabled(True)
        #self.convert2pic.setEnabled(True)
        #self.singal2pic.setEnabled(True)
        self.s2p_thread.quit()
        self.convert_thread.quit()
        self.train_progressBar.setHidden(True)

    def train(self):
        if self.epochs.text() == '' or self.batch_size.text() == '':
            QMessageBox.warning(self, 'warning', '请输入对应数值！')
            return

        # 不存在训练集，则返回
        if not os.path.exists(os.path.join(self.code_root, 'data/train')) or \
                not os.path.exists(os.path.join(self.code_root, 'data/val')) or \
                len(os.listdir(os.path.join(self.code_root, 'data/train'))) == 0 or \
                len(os.listdir(os.path.join(self.code_root, 'data/val'))) == 0:
            QMessageBox.warning(self, 'warning', 'data/train or data/val 不存在或文件夹为空！')
            return

        classes_num = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'num_classes':
                        classes_num = int(item[1])
            if classes_num == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少num_classes数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        if self.save_last_only.isChecked():
            save_last = True
        else:
            save_last = False
        epochs = int(self.epochs.text())
        self.total_epochs = epochs
        bs = int(self.batch_size.text())

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
            f.write(f'model_save_path:{save_dir};  epochs:{epochs};  classes_num:{classes_num};  bath_size:{bs}\n')

        # 初始化train线程
        self.train_thread = Train_thread(root_path=self.code_root, classes_num=classes_num,
                                         epochs=epochs, batch_size=bs, model_save_path=save_dir, save_last=save_last)
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
        self.train_progressBar.setValue(0)
        self.show_tb_btn.setEnabled(True)
        self.draw_thread.start()

        self.trai_button.setEnabled(False)

    def draw_in_time(self, result_tuple):
        acc, loss, val_acc, val_loss = result_tuple[0]
        now_epoch = int(result_tuple[1])
        total_epochs = int(self.total_epochs)
        self.train_progressBar.setValue((now_epoch / total_epochs) * 100)

        self.result_plot.acc_axes.cla()
        self.result_plot.acc_axes.plot(acc, label='Training Acc')
        self.result_plot.acc_axes.plot(val_acc, label='Validation Acc')
        self.result_plot.acc_axes.set_title('sup_acc')
        self.result_plot.acc_axes.set_xlabel('epochs')
        self.result_plot.acc_axes.set_ylabel('acc')
        self.result_plot.acc_axes.legend()

        self.result_plot.loss_axes.cla()
        self.result_plot.loss_axes.plot(loss, label='Training Loss')
        self.result_plot.loss_axes.plot(val_loss, label='Validation Loss')
        self.result_plot.loss_axes.set_title('sup_loss')
        self.result_plot.loss_axes.set_xlabel('epochs')
        self.result_plot.loss_axes.set_ylabel('loss')
        self.result_plot.loss_axes.legend()

        self.result_plot.draw()

    # 绘图同时结束训练
    def draw_result(self, dict, save_path):

        pic_save_path = os.path.join(save_path, 'SUP_result.jpg')
        gif_save_path = os.path.join(save_path, 'SUP_result.gif')
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

        self.trai_button.setEnabled(True)

    def predict(self):

        if self.test_data == '':
            QMessageBox.warning(self, 'warning', '未加载数据！')
            return
        if self.model_path == '':
            QMessageBox.warning(self, 'warning', '未加载模型！')
            return

        classes_num = -1
        data_length = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'num_classes':
                        classes_num = int(item[1])
                    if item[0] == 'data_length':
                        data_length = int(item[1])
            if classes_num == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少num_classes数值')
                return
            if data_length == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少data_length数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        clas, clas_pro = predict(self.model_path, self.test_data, data_length, classes_num, code_root=self.code_root)
        print(clas, clas_pro)

        if os.path.isfile('./data/classes.txt'):
            with open('./data/classes.txt', 'r') as f:
                classes_name = [tmp.strip() for tmp in f.readlines()]
            self.class_txt.setText(str(classes_name[clas]))
        else:
            self.class_txt.setText(str(clas))
        self.class_pro.setText(str(clas_pro))

    def load_tb(self):
        web_site = r'http://127.0.0.1:6006/'
        self.tb_wind = TB_MainWind(web_site)
        self.tb_wind.show()


class Train_thread(QThread):

    sig = pyqtSignal(dict, str)

    def __init__(self, root_path, classes_num, epochs, batch_size, model_save_path, save_last):
        super(Train_thread, self).__init__()
        self.root_path = root_path
        self.classes_num = classes_num
        self.epochs = epochs
        self.bs = batch_size
        self.model_save_path = model_save_path
        self.save_last = save_last

    def run(self):
        result_dict = train(self.root_path, self.classes_num, epochs=self.epochs, batch_size=self.bs,
                            model_save_path=self.model_save_path, save_last=self.save_last)
        self.sig.emit(result_dict, self.model_save_path)


class S2P_thread(QThread):

    sig = pyqtSignal()

    def __init__(self, train_path, val_path, code_path, classes_num):
        super(S2P_thread, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.code_path = code_path
        self.classes_num = classes_num

    def run(self):
        signal2pic(self.train_path, os.path.join(self.code_path, 'data/train'), self.classes_num, self.code_path)
        signal2pic(self.val_path, os.path.join(self.code_path, 'data/val'), self.classes_num, self.code_path)
        self.sig.emit()


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


class Convert_in_time_thread(QThread):
    sig = pyqtSignal(int)

    def __init__(self, code_root):
        super(Convert_in_time_thread, self).__init__()
        self.temp_root = os.path.join(code_root, 'data/data_list.txt')

    def run(self):
        while True:
            if not os.path.isfile(self.temp_root):
                continue
            with open(self.temp_root, 'r') as f:
                dealed_data_len = len(f.readlines())
            self.sig.emit(dealed_data_len)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wind = SUP_Wind()
    wind.show()
    sys.exit(app.exec_())
