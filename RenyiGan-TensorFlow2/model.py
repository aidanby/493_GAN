# Functions to create discriminator and generator  

import tensorflow as tf
from tensorflow.keras import layers

def get_generator():

    inputs = tf.keras.Input(shape=(100,))
    generator = layers.Dense(7*7*256, use_bias=False)(inputs)
    generator = layers.BatchNormalization()(generator)
    generator = layers.LeakyReLU()(generator)
    generator = layers.Reshape((7, 7, 256))(generator)
    generator = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.LeakyReLU()(generator)
    generator = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.LeakyReLU()(generator)
    out = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(generator)

    return tf.keras.Model(inputs=inputs, outputs=out)


def get_discriminator():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



