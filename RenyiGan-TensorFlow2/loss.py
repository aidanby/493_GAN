# Functions to create loss objectives for the discriminator and generator  

import tensorflow as tf


def generator_loss_original(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss_original(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss_renyi(fake_output, alpha):
    f = tf.math.reduce_mean(tf.math.pow(1.0 - fake_output,
                                        (alpha - 1.0) * tf.ones_like(fake_output)))
    loss = 1.0 / (alpha - 1.0) * tf.math.log(f)
    return loss

def generator_loss_renyiL1(fake_output, alpha):
    f = tf.math.reduce_mean(tf.math.pow(1 - fake_output,
                                        (alpha - 1) * tf.ones_like(fake_output)))
    gen_loss = tf.math.abs(1.0 / (alpha - 1) * tf.math.log(f + 1e-8) + tf.math.log(2.0))
    return gen_loss

def discriminator_loss_renyi(real_output, fake_output, alpha):
    f = (tf.math.pow(real_output, alpha) * \
         tf.math.pow(fake_output, (1.0 - alpha)) + \
        tf.math.pow((1.0 - real_output), alpha) * \
        tf.math.pow((1.0 - fake_output), (1.0 - alpha))) / \
        (tf.math.pow(real_output, alpha) +
         tf.math.pow((1.0 - real_output), alpha))
    loss = tf.math.reduce_mean(1.0 / (alpha - 1.0) * tf.math.log(f))
    return loss
