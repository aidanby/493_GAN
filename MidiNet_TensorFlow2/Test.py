import numpy as np 


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from model import get_generator, get_discriminator


G = get_discriminator()
tf.keras.utils.plot_model(G, to_file="Discirminator.png", show_shapes=True)