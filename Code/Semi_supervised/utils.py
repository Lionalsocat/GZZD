import numpy as np
import keras
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image


def loadData(data_path):
    data = np.load(data_path)
    x_train = data[:, :-1]
    y_train = data[:, -1]

    return np.array(x_train), np.array(y_train)


def loadUnlabeled(data_path):
    data = np.load(data_path)

    return data


def loadPicData(pic_path):
    cls = [c for c in os.listdir(pic_path) if os.path.isdir(os.path.join(pic_path, c))]

    k = 0.8
    x_train, y_train = list(), list()
    x_test, y_test = list(), list()
    for c in cls:
        file_list = [f for f in os.listdir(os.path.join(pic_path, c)) if f.endswith('.jpg')]
        for i in range(len(file_list)):
            img = np.array(Image.open(os.path.join(pic_path, c, file_list[i])), dtype=np.float32)
            img /= 255.0
            if i < (len(file_list) * k):
                x_train.append(img)
                y_train.append(int(c))
            else:
                x_test.append(img)
                y_test.append(int(c))

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def loadUnlabeledPicData(pic_path):
    x_train, y_train = list(), list()

    file_list = [f for f in os.listdir(pic_path) if f.endswith('.jpg')]
    for i in range(len(file_list)):
        img = np.array(Image.open(os.path.join(pic_path, file_list[i])), dtype=np.float32)
        img /= 255.0
        x_train.append(img)
        y_train.append(int(-1))

    return np.asarray(x_train), np.asarray(y_train)


def custom_activation(output):
    logexpsum = keras.backend.sum(keras.backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1)
    return result


def select_supervised_samples(dataset, n_samples=70, n_classes=7):
    x, y = dataset
    x_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)

    for i in range(n_classes):
        x_with_class = x[y == i]
        ix = np.random.randint(0, len(x_with_class), n_per_class)

        [x_list.append(x_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]

    return np.asarray(x_list), np.asarray(y_list)


def generate_real_samples(dataset, n_samples):
    signals, labels = dataset

    ix = np.random.randint(0, signals.shape[0], n_samples)
    x, labels = signals[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [x, labels], y


def generate_reals_samples(dataset, n_samples):
    signals = dataset

    ix = np.random.randint(0, signals.shape[0], n_samples)
    x = signals[ix]
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim, n_samples):
    z_input = np.random.randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input


def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)

    signals = generator.predict(z_input)
    y = np.zeros((n_samples, 1))
    return signals, y


def summarize_performance(step, g_model, c_model, dataset, model_save_path):
    x, y = dataset
    _, acc = c_model.evaluate(x, y, verbose=0)
    print(f'Classifier Accuracy: {(acc * 100)}')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    with open(os.path.join(model_save_path, 'runs.txt'), 'a') as f:
        f.write(f'iter:{(step + 1)}, acc:{(acc * 100)} \n')

    c_filename = 'c_model_%04d.h5' % (step + 1)
    c_model.save(os.path.join(model_save_path, c_filename))

    g_filename = 'g_model_%04d.h5' % (step + 1)
    g_model.save(os.path.join(model_save_path, g_filename))
    print(f'>Saved: {c_filename} {g_filename} ')


def signal2pic(dataset, num_classes, root_path):
    x, y = dataset

    saved_path = os.path.join(root_path, 'labeled')
    if os.path.exists(saved_path):
        remove_dir(saved_path)
    os.makedirs(saved_path)

    for i in range(num_classes):
        x_with_class = x[y == i]
        if not os.path.exists(os.path.join(saved_path, str(i))):
            os.makedirs(os.path.join(saved_path, str(i)))
        for j in range(len(x_with_class)):
            my_dpi = 100
            fig = plt.figure(figsize=(416 / my_dpi, 416 / my_dpi), dpi=my_dpi)
            _, _, _, _ = plt.specgram(x_with_class[j], NFFT=128, Fs=1000, noverlap=126,
                                      window=signal.get_window(('kaiser', 18.0), 128))
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(os.path.join(saved_path, str(i), 'signal_' + str(i) + '_' + str(j) + '.jpg'))
            plt.close(fig)
            print('signal_' + str(i) + '_' + str(j) + '.jpg' + '  done!')
            with open(os.path.join(root_path, 'data_list.txt'), 'a') as f:
                f.write(os.path.join(saved_path, str(i), 'signal_' + str(i) + '_' + str(j) + '.jpg') + '\n')


def signal2pic_unlabeled(dataset, root_path):
    x = dataset
    saved_path = os.path.join(root_path, 'unlabeled')
    if os.path.exists(saved_path):
        remove_dir(saved_path)
    os.makedirs(saved_path)

    for i in range(len(x)):
        my_dpi = 100
        fig = plt.figure(figsize=(416 / my_dpi, 416 / my_dpi), dpi=my_dpi)
        _, _, _, _ = plt.specgram(x[i], NFFT=128, Fs=1000, noverlap=126,
                                  window=signal.get_window(('kaiser', 18.0), 128))
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(os.path.join(saved_path, 'signal_' + 'unlabeled' + '_' + str(i) + '.jpg'))
        plt.close(fig)
        print('signal_' + 'unlabeled' + '_' + str(i) + '.jpg' + '  done!')
        with open(os.path.join(root_path, 'data_list.txt'), 'a') as f:
            f.write(os.path.join(saved_path, 'signal_' + 'unlabeled' + '_' + str(i) + '.jpg') + '\n')


def signal2pic_test(dataset, root_path):
    x = dataset

    saved_path = os.path.join(root_path, 'test')
    if os.path.exists(saved_path):
        remove_dir(saved_path)
    os.makedirs(saved_path)

    for i in range(len(x)):
        my_dpi = 100
        fig = plt.figure(figsize=(416 / my_dpi, 416 / my_dpi), dpi=my_dpi)
        _, _, _, _ = plt.specgram(x[i], NFFT=128, Fs=1000, noverlap=126,
                                  window=signal.get_window(('kaiser', 18.0), 128))
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(os.path.join(saved_path, 'test_' + str(i) + '.jpg'))
        plt.close(fig)
        print('test_' + str(i) + '.jpg' + '  done!')


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
