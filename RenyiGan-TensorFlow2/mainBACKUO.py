# Builds and train the DCGAN model  
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import data
import loss
from model import get_generator, get_discriminator, build_generator, build_discriminator




BUFFER_SIZE = 60000
BATCH_SIZE = 100
EPOCHS = 10
test_size = 10000

alpha_g = 0.1
alpha_d = 0.1

version = 1
trial = 1



noise_dim = 28*28
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


(dataset, real_mu, real_sigma) = data.load_mnist(BUFFER_SIZE, BATCH_SIZE) #ADD to train function
generator = build_generator()  #Add to build function
discriminator = build_discriminator() #Add to build function
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)

checkpoint_dir = 'data/renyiganV_' + str(version) + '/AlphaG=' + str(alpha_g)  + '_AlphaD='+ str (alpha_d) + '/trial' + str(trial) + './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

image_dir = 'data/renyiganV_' + str(version) + '/AlphaG=' + str(alpha_g)  + '_AlphaD='+ str (alpha_d) + '/trial' + str(trial) + '/images'
plot_dir = 'data/renyiganV_' + str(version) + '/AlphaG=' + str(alpha_g)  + '_AlphaD='+ str (alpha_d) + '/trial' + str(trial)+ '/plots'






def initalize():
    make_directory('data')
    make_directory('data/renyiganV_' + str(version))
    make_directory('data/renyiganV_' + str(version) + '/AlphaG=' + str(alpha_g)  + '_AlphaD='+ str (alpha_d))
    make_directory('data/renyiganV_' + str(version) + '/AlphaG=' + str(alpha_g)  + '_AlphaD='+ str (alpha_d) + '/trial' + str(trial))
    make_directory(image_dir)
    make_directory(plot_dir)



def make_directory(PATH):
    if not os.path.exists(PATH):
        os.mkdir(PATH)



@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated_images, training=True)

        #gen_loss = loss.generator_loss_original(fake_out)
        #disc_loss = loss.discriminator_loss_original(real_out,fake_out)

        #gen_loss = loss.generator_loss_renyiL1(fake_out, alpha_g)
        disc_loss = loss.discriminator_loss_rgan(real_out,fake_out, alpha_d)
        gen_loss = loss.generator_loss_rgan(fake_out, alpha_g)


        # this is printing all the red numbers and will show 'nan' if broken
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
   # tf.print(disc_loss, gen_loss)
    return disc_loss, gen_loss


def train(dataset, epochs):
    disc_hist, gen_hist, fid_hist = list(), list(), list()
    best_fid = 10000
    best_epoch = 0
    for epoch in range(epochs):
        start = time.time()
        dloss = 0
        gloss = 0
        batchnum = 0
        fid = 0
        for image_batch in dataset:
            batchnum = batchnum + 1
            (d, g) = train_step(image_batch)
            dloss = dloss + d
            gloss = gloss + g
        fid = calculate_fid()

        if (fid < best_fid):
            best_fid = fid
            best_epoch = epoch + 1

        fid_hist.append(fid)
        disc_hist.append(dloss/(batchnum))
        gen_hist.append(gloss/batchnum)


        if (epoch + 10) % 10 == 0:
             checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec. FID: {}'.format(epoch + 1, time.time()-start, fid))
    
        plot_and_save_history(disc_hist, gen_hist, fid_hist, best_fid, best_epoch)
        generate_and_save_images(generator, epoch + 1, seed)




def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig(image_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()


def calculate_fid():
  fake_images = generator(tf.random.normal([test_size, noise_dim])) 
  fake_images = fake_images.numpy()
  fake_images = fake_images.reshape(fake_images.shape[0], 28*28).astype('float32')

  fake_images = (fake_images * 127.5 + 127.5) / 255.0
  fake_mu = fake_images.mean(axis=0)
  fake_sigma = np.cov(np.transpose(fake_images))
  covSqrt = sp.linalg.sqrtm(np.matmul(fake_sigma, real_sigma))
  
  if np.iscomplexobj(covSqrt):
    covSqrt = covSqrt.real
  fidScore = np.linalg.norm(real_mu - fake_mu) + np.trace(real_sigma + fake_sigma - 2 * covSqrt)

  return fidScore


def plot_and_save_history(d_hist, g_hist, fid_hist, best_fid, best_epoch):

    with open(plot_dir + '/History.txt', 'w') as output:
        output.write("FIDScore: " + str(fid_hist) + "\nAvgD D Loss: " + str(np.array(d_hist)) +"\nAvg G Loss: " + str(np.array(g_hist)) \
        + "\n Best FID: " + str(best_fid) + " At Epoch: " + str(best_epoch))
	# plot loss
    plt.figure(1)
    plt.plot(d_hist, label='Discriminator Loss')
    plt.plot(g_hist, label='Generator Loss')
    plt.legend()
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.savefig(plot_dir + '/Loss_History.png')
    plt.close()

    plt.figure(2)

    # plot discriminator accuracy
    plt.plot(fid_hist)
    plt.title("FID History")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.savefig( plot_dir + '/FID_Plot.png')
    plt.close()





initalize()
train(dataset, EPOCHS)






