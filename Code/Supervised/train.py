from Code.Supervised.utils import load_pic_data, signal2pic
from Code.Supervised.model import MobileNetv2
import os
import keras
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def train(root_path, num_classes, epochs, batch_size, model_save_path, save_last):

    # 限制显存，防止报错
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    data_train = load_pic_data(os.path.join(root_path, 'data/train'))
    data_val = load_pic_data(os.path.join(root_path, 'data/val'))

    history_callback = History_callback(model_save_path=model_save_path)

    log_dir = os.path.join(model_save_path, 'logs')
    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')

    if save_last:
        save_model_path = os.path.join(model_save_path, 'model.h5')
    else:
        save_model_path = os.path.join(model_save_path, 'model_epoch_{epoch:03d}.h5')
    check_point = keras.callbacks.ModelCheckpoint(save_model_path)

    model = MobileNetv2((416, 416, 3), num_classes)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=data_train[0], y=data_train[1], epochs=epochs, batch_size=batch_size,
                        validation_data=(data_val[0], data_val[1]), shuffle=True, verbose=1,
                        callbacks=[check_point, history_callback, tb_callback])

    # save_model_fin_path = os.path.join(model_save_path, 'model_fin.h5')
    # model.save(save_model_fin_path)

    return history.history


class History_callback(keras.callbacks.Callback):

    def __init__(self, model_save_path):
        super(History_callback, self).__init__()
        self.model_save_path = model_save_path
        self.log_path = os.path.join(self.model_save_path, 'runs.txt')

    def on_train_begin(self, logs=None):
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)

    def on_epoch_end(self, epoch, logs=None):
        loss_this_epoch = logs.get('loss')
        acc_this_epoch = logs.get('accuracy')
        val_loss_this_epoch = logs.get('val_loss')
        val_acc_this_epoch = logs.get('val_accuracy')

        with open(self.log_path, 'a') as f:
            f.write(f'epoch:{epoch}, loss:{loss_this_epoch}, acc:{acc_this_epoch}, '
                    f'val_loss:{val_loss_this_epoch}, val_acc:{val_acc_this_epoch}\n')
