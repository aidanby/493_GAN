import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from model import get_generator, get_discriminator

# Aidan's code
generator = get_generator()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = get_discriminator()
decision = discriminator(generated_image)
print(decision)


# Jesse's code
# G = get_discriminator()
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# tf.keras.utils.plot_model(G, to_file="Discriminator.png", show_shapes=True)



