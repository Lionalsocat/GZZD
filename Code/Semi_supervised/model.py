import keras
from keras import layers
from keras.models import Model
from Code.Semi_supervised.utils import custom_activation


def define_discriminator(in_shape=(416, 416, 3), n_classes=7):
    in_image = layers.Input(shape=in_shape)

    fe = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = layers.LeakyReLU(alpha=0.2)(fe)

    fe = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)

    fe = layers.Flatten()(fe)

    fe = layers.Dropout(0.4)(fe)
    fe = layers.Dense(n_classes)(fe)

    c_out_layer = layers.Softmax(axis=-1)(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
                    metrics=['accuracy'])

    d_out_layer = layers.Lambda(custom_activation)(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return d_model, c_model


def define_generator(latent_dim):
    in_lat = layers.Input(shape=(latent_dim,))

    n_nodes = 64 * 26 * 26
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((26, 26, 64))(gen)

    gen = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)

    out_layer = layers.Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)

    model = Model(in_lat, out_layer)
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False

    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return model

