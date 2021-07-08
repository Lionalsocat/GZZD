import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image
from matplotlib.animation import FuncAnimation


def signal2pic(data_path, pic_path, num_classes, code_root):
    if os.path.exists(pic_path):
        remove_dir(pic_path)
    os.makedirs(pic_path)

    data = np.load(data_path)
    x = data[:, :-1]
    y = data[:, -1]

    for i in range(num_classes):
        x_with_class = x[y == i]
        if not os.path.exists(os.path.join(pic_path, str(i))):
            os.makedirs(os.path.join(pic_path, str(i)))
        for j in range(len(x_with_class)):
            my_dpi = 100
            fig = plt.figure(figsize=(416 / my_dpi, 416 / my_dpi), dpi=my_dpi)
            _, _, _, _ = plt.specgram(x_with_class[j], NFFT=128, Fs=1000, noverlap=126,
                                      window=signal.get_window(('kaiser', 18.0), 128))
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(os.path.join(pic_path, str(i), 'signal_' + str(i) + '_' + str(j) + '.jpg'))
            plt.close(fig)
            print('signal_' + str(i) + '_' + str(j) + '.jpg' + '  done!')
            with open(os.path.join(code_root, 'data/data_list.txt'), 'a') as f:
                f.write(os.path.join(pic_path, str(i), 'signal_' + str(i) + '_' + str(j) + '.jpg') + '\n')


def signal2pic_test(dataset, save_path):
    x = dataset

    if os.path.exists(save_path):
        remove_dir(save_path)
    os.makedirs(save_path)

    for i in range(len(x)):
        my_dpi = 100
        fig = plt.figure(figsize=(416 / my_dpi, 416 / my_dpi), dpi=my_dpi)
        _, _, _, _ = plt.specgram(x[i], NFFT=128, Fs=1000, noverlap=126,
                                  window=signal.get_window(('kaiser', 18.0), 128))
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(os.path.join(save_path, 'test_' + str(i) + '.jpg'))
        plt.close(fig)
        print('test_' + str(i) + '.jpg' + '  done!')


def load_pic_data(pic_path):
    classes = [cls for cls in os.listdir(pic_path) if os.path.isdir(os.path.join(pic_path, cls))]

    x, y = list(), list()
    for cls in range(len(classes)):
        img_files = [file for file in os.listdir(os.path.join(pic_path, classes[cls])) if file.endswith('.jpg')]
        for img_file in img_files:
            img = np.array(Image.open(os.path.join(pic_path, classes[cls], img_file)), dtype=np.float32)
            img /= 255.0
            x.append(img)
            y.append(cls)

    return np.asarray(x), np.asarray(y)


def remove_dir(path):
    if not os.path.isdir(path):
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            remove_dir(item_path)
        elif os.path.isfile(item_path):
            try:
                os.remove(item_path)
            except:
                pass
    try:
        os.rmdir(path)
    except:
        pass


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
