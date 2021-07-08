# 导入相关库
from Code.Unsupervised import sae as SAE
import numpy as np
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Code.suff_deal import get_acc_loss


# 训练函数
def train(dataset_path, class_dim, ori_dim, batch_size, epochs_layer, epochs_whole, model_save_path,
          save_last, regula=0.05):

    train_path = os.path.join(dataset_path, 'data_train.npy')
    val_path = os.path.join(dataset_path, 'data_val.npy')

    x_train = np.load(train_path)[:, :-1]
    x_train = deal_x(x_train)
    y_train = np.array(to_categorical(list(np.load(train_path)[:, -1])))
    x_test = np.load(val_path)[:, :-1]
    x_test = deal_x(x_test)
    y_test = np.array(to_categorical(list(np.load(val_path)[:, -1])))

    # 参数设置
    class_dim = class_dim
    batch_size = batch_size
    model_save_path = model_save_path
    epochs_layer = epochs_layer
    epochs_whole = epochs_whole
    regula = regula
    origin_dim = ori_dim
    h_dim1 = ori_dim//2
    h_dim2 = 64
    h_dim3 = 32
    h_dim4 = 16

    # 5层的stacked ae，实际上要使用4个ae，实例化4个ae
    # 参数：SAE.AutoEncoderLayer(input_dim, output_dim, regula=0.05)
    num_layers = 5
    encoder_1 = SAE.AutoEncoderLayer(origin_dim, h_dim1, regula=regula)
    encoder_2 = SAE.AutoEncoderLayer(h_dim1, h_dim2, regula=regula)
    decoder_3 = SAE.AutoEncoderLayer(h_dim2, h_dim3, regula=regula)
    decoder_4 = SAE.AutoEncoderLayer(h_dim3, h_dim4, regula=regula)
    autoencoder_list = [encoder_1, encoder_2, decoder_3, decoder_4]

    # 按照顺序对每一层进行预训练
    # 参数：SAE.train_layers(x_train, encoder_list, layer, epochs, batch_size)
    print("Pre training:")
    for level in range(num_layers - 1):
        print("level:", level)
        SAE.train_layers(x_train=x_train, encoder_list=autoencoder_list, layer=level, epochs=epochs_layer,
                         batch_size=batch_size)

    # 用训练好的4个ae构建stacked ae
    # 参数：SAE.StackedAutoEncoder(layer_list, num_layers, class_dim, regula=0.05)
    stacked_ae = SAE.StackedAutoEncoder(autoencoder_list, num_layers=num_layers, class_dim=class_dim)

    # 进行全局训练优化,训练分类器
    # 参数：train_whole(dataset, sae, epochs, batch_size, save_dir='./model/', save_name='model.h5')
    print("Whole training:")
    result_dict = SAE.train_whole(dataset=(x_train, y_train, x_test, y_test), sae=stacked_ae, epochs=epochs_whole,
                                  batch_size=batch_size, save_dir=model_save_path, save_last=save_last)

    return result_dict


def deal_x(x):
    return_x = []
    for i in range(len(x)):
        temp_x = x[i]
        new_x = np.fft.fft(temp_x)
        new_x_abs = np.fft.fftshift(abs(new_x))
        return_x.append(new_x_abs)
    return np.array(return_x)


def draw_acc_loss(result_dict, path):
    acc = result_dict['accuracy']
    val_acc = result_dict['val_accuracy']
    loss = result_dict['loss']
    val_loss = result_dict['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Acc')
    plt.plot(val_acc, label='Validation Acc')
    plt.title('Acc', fontproperties='SimHei')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss', fontproperties='SimHei')
    plt.legend()
    # plt.show()
    plt.savefig(path)


class draw_acc_loss_ani:
    def __init__(self, result_dict, save_path):
        self.acc = result_dict['accuracy']
        self.val_acc = result_dict['val_accuracy']
        self.loss = result_dict['loss']
        self.val_loss = result_dict['val_loss']
        self.fig, self.axes = plt.subplots(1, 2)

        ani = FuncAnimation(self.fig, self.update_ani, frames=len(self.acc), interval=500)
        ani.save(save_path)

    def update_ani(self, i):

        acc_temp = self.acc[:i+1]
        val_acc_temp = self.val_acc[:i+1]
        loss_temp = self.loss[:i+1]
        val_loss_temp = self.val_loss[:i+1]

        self.axes[0].cla()
        self.axes[0].plot(acc_temp, label='Training Acc')
        self.axes[0].plot(val_acc_temp, label='Validation Acc')
        self.axes[0].set_title('Accuracy')
        self.axes[0].set_xlabel('epochs')
        self.axes[0].set_ylabel('acc')
        self.axes[0].legend()

        self.axes[1].cla()
        self.axes[1].plot(loss_temp, label='Training Loss')
        self.axes[1].plot(val_loss_temp, label='Validation Loss')
        self.axes[1].set_title('Loss')
        self.axes[1].set_xlabel('epochs')
        self.axes[1].set_ylabel('loss')
        self.axes[1].legend()

