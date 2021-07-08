from PyQt5.Qt import *
from UI.SEMISUP_ui import Ui_MainWindow
from TB_main import TB_MainWind
import sys
from Code.Semi_supervised.train import train_sgan, draw_acc, draw_acc_ani
from Code.Semi_supervised.predict import predict
from Code.Semi_supervised.utils import *
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


class SEMISUP_Wind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.label_data = ''
        self.unlabel_data = ''
        self.data_test_path = ''
        self.model_path = ''

        self.code_root = './Code/Semi_supervised'
        self.model_save_path = './result/Semi_supervised'

        self.result_plot = Result_Canvas(width=5, height=3, dpi=70)
        self.result_plot_ntb = NavigationToolBar(self.result_plot, self)
        self.intime_layout.addWidget(self.result_plot)
        self.intime_layout.addWidget(self.result_plot_ntb)

        self.singal2pic.stateChanged.connect(self.s2pchanged)

        self.train_progressBar.setValue(0)
        self.train_progressBar.setHidden(True)

        self.load_labeled.clicked.connect(self.load_label_data)
        self.load_unlabeled.clicked.connect(self.load_unlabel_data)
        self.trai_button.clicked.connect(self.train)
        self.load_test_b.clicked.connect(self.load_test)
        self.model_b.clicked.connect(self.load_model)
        self.predict_b.clicked.connect(self.predict)
        self.show_tb_btn.clicked.connect(self.load_tb)
        self.convert2pic.clicked.connect(self.s2p_start)

    def s2pchanged(self):
        self.convert2pic.setEnabled(self.singal2pic.isChecked())

    def s2p_start(self):

        if self.label_data == '' or self.unlabel_data == '':
            QMessageBox.warning(self, 'warning', '未生成pic数据集， 则需要手动指定数据！')
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

        data = loadData(self.label_data)
        self.data_n = len(data[0])
        unlabeled = loadUnlabeled(self.unlabel_data)
        self.unlabeld_n = len(unlabeled)

        self.s2p_thread = S2P_thread(labeled_data=data, unlabeled_data=unlabeled,
                                     classes_num=classes_num, code_path=self.code_root)
        self.s2p_thread.sig.connect(self.s2p_finish)

        if os.path.isfile(os.path.join(self.code_root, 'data_list.txt')):
            os.remove(os.path.join(self.code_root, 'data_list.txt'))

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
        self.train_progressBar.setValue((dealed_len/(self.data_n+self.unlabeld_n))*100)

    def s2p_finish(self):
        QMessageBox.information(self, 'INFO', '转换完成!')
        self.trai_button.setEnabled(True)
        self.convert2pic.setEnabled(True)
        self.singal2pic.setEnabled(True)
        self.s2p_thread.quit()
        self.convert_thread.change_status()
        self.convert_thread.quit()
        self.train_progressBar.setHidden(True)

    def load_label_data(self):
        self.label_data, _ = QFileDialog.getOpenFileName(self, "加载有标签数据", './data', "NPY Files(*.npy)")

    def load_unlabel_data(self):
        self.unlabel_data, _ = QFileDialog.getOpenFileName(self, "加载无标签数据", './data', "NPY Files(*.npy)")

    def train(self):
        if self.epochs.text() == '' or self.batch_size.text() == '' or self.latent_dim.text() == '':
            QMessageBox.warning(self, 'warning', '请输入对应数值!')
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

        # 不存在训练集，则返回
        if not os.path.exists(os.path.join(self.code_root, 'labeled')) or \
                not os.path.exists(os.path.join(self.code_root, 'unlabeled')) or\
                len(os.listdir(os.path.join(self.code_root, 'labeled'))) == 0 or\
                len(os.listdir(os.path.join(self.code_root, 'unlabeled'))) == 0:
            QMessageBox.warning(self, 'warning', 'labeled/unlabeled 不存在或者文件夹为空！')
            return

        if self.save_last_only.isChecked():
            save_last = True
        else:
            save_last = False
        latent_dim = int(self.latent_dim.text())
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
            f.write(f'model_save_path:{save_dir};  epochs:{epochs};  classes_num:{classes_num};'
                    f'  latent_dim:{latent_dim};  batch_size:{bs}\n')

        # 初始化train线程
        self.train_thread = Train_thread(data_root=self.code_root, classes_num=classes_num,
                                         latent_dim=latent_dim, epochs=epochs, batch_size=bs,
                                         model_save_path=save_dir, save_last=save_last)
        self.train_thread.sig.connect(self.draw_result)

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
        self.convert2pic.setEnabled(False)
        self.singal2pic.setEnabled(False)
        self.draw_thread.start()

        self.trai_button.setEnabled(False)

    def draw_in_time(self, result_tuple):
        acc, loss, val_acc, val_loss = result_tuple[0]
        now_epoch = int(result_tuple[1])
        total_epochs = int(self.total_epochs)
        self.train_progressBar.setValue((now_epoch/total_epochs)*100)

        self.result_plot.acc_axes.cla()
        self.result_plot.acc_axes.plot(acc, label='Training Acc')
        self.result_plot.acc_axes.plot(val_acc, label='Validation Acc')
        self.result_plot.acc_axes.set_title('semi_sup_acc')
        self.result_plot.acc_axes.set_xlabel('epochs')
        self.result_plot.acc_axes.set_ylabel('acc')
        self.result_plot.acc_axes.legend()

        self.result_plot.loss_axes.cla()
        self.result_plot.loss_axes.plot(loss, label='Training Loss')
        self.result_plot.loss_axes.plot(val_loss, label='Validation Loss')
        self.result_plot.loss_axes.set_title('semi_sup_loss')
        self.result_plot.loss_axes.set_xlabel('epochs')
        self.result_plot.loss_axes.set_ylabel('loss')
        self.result_plot.loss_axes.legend()

        self.result_plot.draw()

    def draw_result(self, path, save_path):

        jpg_save_path = os.path.join(save_path, 'SEMISUP_result.jpg')
        gif_save_path = os.path.join(save_path, 'SEMISUP_result.gif')
        draw_acc(path, jpg_save_path)
        draw_acc_ani(path, gif_save_path)
        QMessageBox.information(self, 'INFO', '训练完成!')
        # 退出线程
        self.train_thread.quit()
        self.draw_thread.quit()
        self.draw_thread.change_status()
        self.train_progressBar.setHidden(True)
        self.show_tb_btn.setEnabled(False)
        self.convert2pic.setEnabled(self.convert2pic.setEnabled(self.singal2pic.isChecked()))
        self.singal2pic.setEnabled(True)
        # 结束tensorboard
        self.pipe.terminate()

        self.trai_button.setEnabled(True)

    def load_test(self):
        self.data_test_path, _ = QFileDialog.getOpenFileName(self, "加载测试数据", './', "MAT Files(*.mat)")

    def load_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, "加载模型", self.model_save_path, "H5 Files(*.h5)")

    def predict(self):
        if self.data_test_path == '':
            QMessageBox.warning(self, 'warning', '未加载数据！')
            return
        if self.model_path == '':
            QMessageBox.warning(self, 'warning', '未加载模型！')
            return

        data_length = -1
        if os.path.isfile('./data/data_info.txt'):
            with open('./data/data_info.txt', 'r') as f:
                items = [item.strip().split(':') for item in f.readlines()]
                for item in items:
                    if item[0] == 'data_length':
                        data_length = int(item[1])
            if data_length == -1:
                QMessageBox.warning(self, 'warning', 'data/data_info.txt中缺少data_length数值')
                return
        else:
            QMessageBox.warning(self, 'warning', 'data/data_info.txt  不存在！')
            return

        clas, clas_pro = predict(self.model_path, self.data_test_path, data_length, root_path=self.code_root)
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

    sig = pyqtSignal(str, str)

    def __init__(self, data_root, classes_num, latent_dim, epochs, batch_size, model_save_path, save_last):
        super(Train_thread, self).__init__()
        self.data_root = data_root
        self.classes_num = classes_num
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.bs = batch_size
        self.model_save_path = model_save_path
        self.save_last = save_last

    def run(self):
        train_sgan(self.data_root, classes_num=self.classes_num, batchsize=self.bs,
                   latent_dim=self.latent_dim, epochs=self.epochs, model_save_path=self.model_save_path,
                   save_last=self.save_last)
        self.sig.emit(os.path.join(self.model_save_path, 'runs.txt'), self.model_save_path)


class S2P_thread(QThread):

    sig = pyqtSignal()

    def __init__(self, labeled_data, unlabeled_data, code_path, classes_num):
        super(S2P_thread, self).__init__()
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.code_path = code_path
        self.classes_num = classes_num

    def run(self):
        signal2pic(self.labeled_data, self.classes_num, self.code_path)
        signal2pic_unlabeled(self.unlabeled_data, self.code_path)
        self.sig.emit()


class Draw_in_time_thread(QThread):
    sig = pyqtSignal(tuple)

    def __init__(self, runs_root):
        super(Draw_in_time_thread, self).__init__()
        self.runs_root = runs_root
        self.is_draw = True
        self.record_len = 0

    def run(self):
        while self.is_draw:
            if not os.path.exists(self.runs_root):
                continue
            record = get_acc_loss(self.runs_root)
            if len(record[0]) > self.record_len:
                self.record_len = len(record[0])
                self.sig.emit((record, self.record_len))

    def change_status(self):
        self.is_draw = False


class Convert_in_time_thread(QThread):
    sig = pyqtSignal(int)

    def __init__(self, code_root):
        super(Convert_in_time_thread, self).__init__()
        self.temp_root = os.path.join(code_root, 'data_list.txt')
        self.is_convert = True

    def run(self):
        while self.is_convert:
            if not os.path.isfile(self.temp_root):
                continue
            with open(self.temp_root, 'r') as f:
                dealed_data_len = len(f.readlines())
            self.sig.emit(dealed_data_len)

    def change_status(self):
        self.is_convert = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sgan_ui = SEMISUP_Wind()
    sgan_ui.show()
    sys.exit(app.exec_())
