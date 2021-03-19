import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import data
import loss
from model import get_generator, get_discriminator, build_generator, build_discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# real_output = tf.constant([[1],[2],[3]], dtype=tf.float32)
# fake_output = tf.constant([[2],[4],[6]], dtype=tf.float32)

# alpha = 2

# n1 = (tf.math.pow(real_output, alpha * tf.ones_like(real_output)))*(tf.math.pow(fake_output, 1.0 - alpha * tf.ones_like(real_output)))
# n2 = (tf.math.pow(1-real_output, alpha * tf.ones_like(real_output)))*(tf.math.pow(1-fake_output, 1.0 - alpha * tf.ones_like(real_output)))
# n = n1 + n2

# d1 = tf.math.pow(real_output, alpha * tf.ones_like(real_output))
# d2 = tf.math.pow(1-real_output, alpha * tf.ones_like(real_output))
# d = d1 + d2

# f = tf.math.reduce_mean(((1/(alpha-1))*tf.math.log(n/d)))


# print(d1)
# print(d2)
# print(d)
# print(f)


disc_hist, gen_hist, fid_hist = list(), list(), list()

disc_hist.append(3)
disc_hist.append(2)
disc_hist.append(1)
gen_hist.append(1)
gen_hist.append(2)
gen_hist.append(3)
fid_hist.append(1)
fid_hist.append(2)
fid_hist.append(1)

out_str = ', ' 
with open('History.txt', 'w') as output:
    output.write("FIDScore: " + str(fid_hist) + '\nHELLO')


def plot_history(d_hist, g_hist, fid_hist):
	# plot loss
    plt.figure(1)
    plt.plot(d_hist, label='Discriminator Loss')
    plt.plot(g_hist, label='Generator Loss')
    plt.legend()
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.savefig('Loss_History.png')
    plt.close()

    plt.figure(2)

    # plot discriminator accuracy
    plt.plot(fid_hist)
    plt.title("FID History")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.savefig( 'FID_Plot.png')
    plt.close()

#tf.math.reduce_mean(tf.math.pow(real_output, alpha * tf.ones_like(fake_output)) ) 


