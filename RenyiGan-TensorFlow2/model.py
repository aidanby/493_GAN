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

# HIMESH MODELS  
# much more stable

def build_generator():
        with tf.name_scope('generator') as scope:
            model = Sequential(name=scope)
            model.add(Dense(7 * 7 * 256, use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01), input_shape=(28 * 28,)))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

            model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False,
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 28, 28, 1)

            return model

def build_discriminator():
        with tf.name_scope('discriminator') as scope:
            model = Sequential(name=scope)
            model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))

            return model

