# Midi Net Discriminator and Generator Model

import tensorflow as tf
from tensorflow.keras import layers

def get_generator():

    inputs = tf.keras.Input(shape=(100,))
    generator = layers.Dense(32*4*256, use_bias=False)(inputs)
    # generator  = layers.Dense(1024)(inputs)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)

    # Not necessary - aidan
    # generator = layers.Dense(256)(generator)
    # generator = layers.BatchNormalization()(generator)
    # generator = layers.ReLU()(generator)

    generator = layers.Reshape([32, 4, 256])(generator)

    generator = layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False)(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    # image size 32 4 128

    generator = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(generator)
    generator = layers.BatchNormalization()(generator)
    generator = layers.ReLU()(generator)
    # image size 64 8 64

    generator = layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False)(generator)
    # image size 128 16 1

    #generator = tf.keras.activations.sigmoid(generator)(generator)
    return tf.keras.Model(inputs=inputs, outputs=generator)


def get_discriminator():
    inputs = tf.keras.Input(shape=(128, 16, 1))

    discriminator = layers.Conv2D(64, (5, 5), strides=(2,2),  padding='VALID')(inputs)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)

    discriminator = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)

    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dense(1)(discriminator)

    return tf.keras.Model(inputs=inputs, outputs=discriminator)

