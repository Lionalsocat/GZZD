"""
构建SAE模型
堆叠自编码器(Stacked AutoEncoder)
"""

# 导入相关库
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import TensorBoard
import keras
import os


# 构建单个autoencoder
class AutoEncoderLayer():
    def __init__(self, input_dim, output_dim, regula=0.05):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regula = regula
        self.build()

    def build(self):
        self.input = Input(shape=(self.input_dim,))
        self.encode_layer = Dense(self.output_dim, activation='relu', kernel_regularizer=regularizers.l2(self.regula))
        self.encoded = self.encode_layer(self.input)
        self.encoder = Model(self.input, self.encoded)

        self.decode_layer = Dense(self.input_dim, activation='relu', kernel_regularizer=regularizers.l2(self.regula))
        self.decoded = self.decode_layer(self.encoded)

        self.autoencoder = Model(self.input, self.decoded)


# 构建堆叠AE(SAE)
class StackedAutoEncoder():
    def __init__(self, layer_list, num_layers, class_dim, regula=0.05):
        self.layer_list = layer_list
        self.regula = regula
        self.num_layers = num_layers
        self.class_dim = class_dim
        self.build()

    def build(self):
        out = self.layer_list[0].encoded
        for i in range(1, self.num_layers - 1):
            out = self.layer_list[i].encode_layer(out)
        outt_layer = Dense(self.class_dim, activation='softmax', kernel_regularizer=regularizers.l2(self.regula))(out)
        self.model = Model(self.layer_list[0].input, outt_layer)


# 训练AE模型
def train_layers(x_train, encoder_list, layer, epochs, batch_size):

    # 对前(layer-1)层用已经训练好的参数进行前向计算，ps:第0层没有前置层
    out = x_train
    origin = x_train
    if layer != 0:
        for i in range(layer):
            out = encoder_list[i].encoder.predict(out)

    encoder_list[layer].autoencoder.summary()
    encoder_list[layer].autoencoder.compile(optimizer='adam', loss='mae')

    # 训练第layer个ae
    encoder_list[layer].autoencoder.fit(
        out,
        origin if layer == 0 else out,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2
    )


# 训练整个模型以及分类器
def train_whole(dataset, sae, epochs, batch_size, save_dir, save_last):

    history_callback = History_callback(model_save_path=save_dir)

    log_dir = os.path.join(save_dir, 'logs')
    tb_callback = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

    if save_last:
        save_model_path = os.path.join(save_dir, 'model.h5')
    else:
        save_model_path = os.path.join(save_dir, 'model_epoch_{epoch:03d}.h5')
    check_point = keras.callbacks.ModelCheckpoint(save_model_path)

    # 开始训练
    x_train, y_train, x_test, y_test = dataset
    sae.model.summary()
    sae.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    history = sae.model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, y_test),
        verbose=2,
        callbacks=[history_callback, tb_callback, check_point]
    )

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
