# Loads and does prepossing on the desired training set 

import tensorflow as tf
import string
import numpy as np
import matplotlib.pyplot as plt
import os



def load_mnist(BUFFER_SIZE, BATCH_SIZE):
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

