import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import data
import loss
from model import get_generator, get_discriminator, build_generator, build_discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

real_output = tf.constant([[1],[2],[3]], dtype=tf.float32)
fake_output = tf.constant([[2],[4],[6]], dtype=tf.float32)

alpha = 2

n1 = (tf.math.pow(real_output, alpha * tf.ones_like(real_output)))*(tf.math.pow(fake_output, 1.0 - alpha * tf.ones_like(real_output)))
n2 = (tf.math.pow(1-real_output, alpha * tf.ones_like(real_output)))*(tf.math.pow(1-fake_output, 1.0 - alpha * tf.ones_like(real_output)))
n = n1 + n2

d1 = tf.math.pow(real_output, alpha * tf.ones_like(real_output))
d2 = tf.math.pow(1-real_output, alpha * tf.ones_like(real_output))
d = d1 + d2

f = tf.math.reduce_mean(((1/(alpha-1))*tf.math.log(n/d)))


print(d1)
print(d2)
print(d)
print(f)

#tf.math.reduce_mean(tf.math.pow(real_output, alpha * tf.ones_like(fake_output)) ) 


