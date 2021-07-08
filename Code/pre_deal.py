import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt


def deal_data(data_path, data_long, overlap, val_pre, unlabel_pre, plot_dis, is_shi):
    # 获得类别名
    classes = [clas for clas in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, clas))]
    # 写出类别文件
    if not os.path.exists('./data'):
        os.makedirs('./data')
    with open('./data/classes.txt', 'w') as f:
        for clas in classes:
            f.write(str(clas) + '\n')
    # 写出信息文件
    with open('./data/data_info.txt', 'w') as f:
        f.write('data_length:' + str(data_long) + '\n')
        f.write('num_classes:' + str(len(classes)) + '\n')
        f.write('data_path:' + str(data_path) + '\n')
        f.write('overlap:' + str(overlap/100) + '\n')
        f.write('val_pre:' + str(val_pre/100) + '\n')
        f.write('unlabel_pre:' + str(unlabel_pre/100) + '\n')

    # 分别打开.mat文件进行数据处理
    x, y = [], []
    for i in range(len(classes)):
        cur_data_path = os.path.join(data_path, classes[i])
        files = [file for file in os.listdir(cur_data_path) if file.endswith('.mat')]
        for file in files:
            file_path = os.path.join(cur_data_path, file)
            data = scio.loadmat(file_path)
            x_before = data['x'].T
            j = 0
            while j < (x_before.size - data_long):
                x.append(x_before[0, j:j + data_long])
                y.append(i)
                j += int((1 - overlap/100) * data_long)
    # 若是频域信号，则先逆fft变回时域信号
    if not is_shi:
        # 默认axis=-1 即按列做ifft
        x = np.fft.ifft(x)
    x, y = np.array(x), np.array(y)

    # 数据分布
    data_distribution = []
    for i in range(len(classes)):
        data_distribution.append((y == i).sum())

    # 构建 data_train.npy 与 data_val.npy
    data_train, data_val = [], []
    for i in range(len(classes)):
        temp_x, temp_y = x[y == i], y[y == i]
        temp_y = temp_y.reshape(temp_y.size, 1)
        temp_data = np.concatenate((temp_x, temp_y), axis=1)
        np.random.shuffle(temp_data)
        val_data = temp_data[:int(len(temp_data)*val_pre/100), :]
        train_data = temp_data[int(len(temp_data)*val_pre/100):, :]
        data_train += list(train_data)
        data_val += list(val_data)
    data_train, data_val = np.array(data_train), np.array(data_val)
    np.save('./data/data_train.npy', data_train)
    np.save('./data/data_val.npy', data_val)

    # 构建 data_labeled.npy 与 data_unlabeled.npy
    data_labeled, data_unlabeled = [], []
    for i in range(len(classes)):
        temp_x, temp_y = x[y == i], y[y == i]
        temp_y = temp_y.reshape(temp_y.size, 1)
        temp_data = np.concatenate((temp_x, temp_y), axis=1)
        np.random.shuffle(temp_data)
        unlabeled_data = temp_data[:int(len(temp_data) * unlabel_pre / 100), :]
        labeled_data = temp_data[int(len(temp_data) * unlabel_pre / 100):, :]
        data_labeled += list(labeled_data)
        data_unlabeled += list(unlabeled_data)
    data_labeled, data_unlabeled = np.array(data_labeled), np.array(data_unlabeled)[:, :-1]
    np.save('./data/data_labeled.npy', data_labeled)
    np.save('./data/data_unlabeled.npy', data_unlabeled)

    # 绘制distribution分布图
    if plot_dis:
        plt.figure(figsize=(8, 8))
        plt.bar(range(len(classes)), data_distribution, align='center')
        plt.xticks(range(len(classes)), classes, fontsize=12, rotation=45)
        for i, v in enumerate(data_distribution):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('class')
        plt.ylabel('number of data')
        plt.title('class distribution')
        plt.savefig('./data/Distribution.jpg')
        plt.close()

    return classes, data_distribution


def draw_pic(data_path, is_shi):
    data = scio.loadmat(data_path)
    x = data['x'][:, 0]
    if not is_shi:
        x = np.fft.ifft(x)
    N = len(x)

    plt.figure(figsize=(10, 8))

    # 绘制时域图
    plt.subplot(211)
    t = np.arange(0, N)
    plt.plot(t, x)
    plt.title('时域图', fontproperties='SimHei')
    plt.axis('tight')

    # 绘制频域图
    plt.subplot(212)
    tf = np.linspace(-N/2, N/2, N)
    xf = np.fft.fft(x)
    xf_abs = np.fft.fftshift(abs(xf)/N)
    plt.plot(tf, xf_abs)
    plt.title('频域图', fontproperties='SimHei')
    plt.axis('tight')

    # plt.show()
    plt.savefig(data_path.replace('mat', 'jpg'))


