# data.py

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
 

def load_data(trainingData):

    BUFFER_SIZE = 4816
    BATCH_SIZE = 256
    train_dataset = np.load(trainingData) #Shape: (n, 1, 16, 128), where n is the number of measures(bars) of training data.
    
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = np.transpose(train_dataset,(0,2,3,1))
    print(train_dataset.shape)
    return train_dataset


load_data('data_x.npy')