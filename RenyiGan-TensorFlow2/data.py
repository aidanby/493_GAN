# Loads and does prepossing on the desired training set 

import tensorflow as tf
import string
import numpy as np
import matplotlib.pyplot as plt
import os



def load_mnist(BUFFER_SIZE, BATCH_SIZE):
    (train_i, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_i.reshape(train_i.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    # test_size = 1000
    # train_d = train_images[:test_size] 
    train_d = train_i.reshape(train_i.shape[0], 28 * 28).astype('float32')
    train_d = train_d / 255.0
    real_mu = train_d.mean(axis=0)
    train_d = np.transpose(train_d)
    real_sigma = np.cov(train_d)


    return train_dataset, real_mu, real_sigma

