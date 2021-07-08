from Code.Semi_supervised.model import *
from Code.Semi_supervised.utils import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import os

from Code.suff_deal import get_acc_loss


def train(g_model, d_model, c_model, gan_model, train_dataset, test_dataset, unsupervised_dataset,
          latent_dim, n_epochs, n_batch, model_save_path, save_last):

    x_sup, y_sup = train_dataset
    print(x_sup.shape, y_sup.shape)

    log_dir = os.path.join(model_save_path, 'logs', 'GCD')
    writer = tf.summary.create_file_writer(log_dir)
    log_dir = os.path.join(model_save_path, 'logs', 'train')
    train_writer = tf.summary.create_file_writer(log_dir)
    log_dir = os.path.join(model_save_path, 'logs', 'validation')
    validation_writer = tf.summary.create_file_writer(log_dir)

    dataset_all = np.vstack((train_dataset[0], unsupervised_dataset[0]))

    bat_per_epo = int(dataset_all[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    print(f'n_epochs = {n_epochs}, n_batch = {n_batch}, 1/2 = {half_batch}, b/e = {bat_per_epo}, steps = {n_steps}')

    for i in range(n_steps):

        [xsup_real, ysup_real], _ = generate_real_samples([x_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(xsup_real, ysup_real)

        x_real, y_real = generate_reals_samples(dataset_all, half_batch)
        d_loss1 = d_model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)

        x_gan, y_gan = generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(x_gan, y_gan)

        with writer.as_default():
            tf.summary.scalar('c_loss', c_loss, i+1)
            tf.summary.scalar('c_acc', c_acc, i+1)
            tf.summary.scalar('d_loss1', d_loss1, i+1)
            tf.summary.scalar('d_loss2', d_loss2, i+1)
            tf.summary.scalar('g_loss', g_loss, i+1)

        print(f'iter:{(i+1)}, c_loss:{c_loss}, c_acc:{c_acc}, d_loss1:{d_loss1}, d_loss2:{d_loss2}, g_loss:{g_loss}')

        if (i + 1) % (bat_per_epo * 1) == 0:

            train_loss, train_acc = c_model.evaluate(x_sup, y_sup, verbose=0)

            x, y = test_dataset
            val_loss, val_acc = c_model.evaluate(x, y, verbose=0)
            print(f'Classifier Accuracy: {(val_acc * 100)}')

            with open(os.path.join(model_save_path, 'runs.txt'), 'a') as f:
                f.write(f'epoch:{(i + 1) / (bat_per_epo * 1)}, loss:{train_loss}, acc:{train_acc}, '
                        f'val_loss:{val_loss}, val_acc:{val_acc}\n')

            with train_writer.as_default():
                tf.summary.scalar('epoch_accuracy', train_acc, step=int((i + 1) / (bat_per_epo * 1)))
                tf.summary.scalar('epoch_loss', train_loss, step=int((i + 1) / (bat_per_epo * 1)))
            with validation_writer.as_default():
                tf.summary.scalar('epoch_accuracy', val_acc, step=int((i + 1) / (bat_per_epo * 1)))
                tf.summary.scalar('epoch_loss', val_loss, step=int((i + 1) / (bat_per_epo * 1)))

            # c_filename = 'model_%04d.h5' % (i + 1)
            if save_last:
                c_filename = 'model.h5'
            else:
                c_filename = 'model_epoch_{:03d}.h5'.format(int((i + 1) / (bat_per_epo * 1)))
            c_model.save(os.path.join(model_save_path, c_filename))

            # 生成器模型
            # g_filename = 'g_model_%04d.h5' % (i + 1)
            # g_model.save(os.path.join(model_save_path, g_filename))
            print(f'>Saved: {c_filename} -->{i+1}  ')


def train_sgan(data_root, classes_num, latent_dim, epochs, batchsize, model_save_path, save_last):

    labeled_path = os.path.join(data_root, 'labeled')
    unlabeled_path = os.path.join(data_root, 'unlabeled')

    d_model, c_model = define_discriminator(n_classes=classes_num)
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)

    dataset = loadPicData(labeled_path)
    train_dataset = dataset[0], dataset[1]
    test_dataset = dataset[2], dataset[3]
    unsupervised_dataset = loadUnlabeledPicData(unlabeled_path)
    train(g_model, d_model, c_model, gan_model, train_dataset, test_dataset, unsupervised_dataset,
          latent_dim, epochs, batchsize, model_save_path, save_last)


def draw_acc(path, save_path):
    acc, loss, val_acc, val_loss = get_acc_loss(path)

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
    plt.savefig(save_path)


class draw_acc_ani:
    def __init__(self, path, save_path):
        self.acc, self.loss, self.val_acc, self.val_loss = get_acc_loss(path)
        self.fig, self.axes = plt.subplots(1, 2)

        ani = FuncAnimation(self.fig, self.update_ani, frames=len(self.acc), interval=500)
        ani.save(save_path)

    def update_ani(self, i):
        acc_temp = self.acc[:i + 1]
        val_acc_temp = self.val_acc[:i + 1]
        loss_temp = self.loss[:i + 1]
        val_loss_temp = self.val_loss[:i + 1]

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


# if __name__ == '__main__':
#     draw_acc_ani('./runs.txt', '1.gif')
