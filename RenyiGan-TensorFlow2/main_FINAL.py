import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import data
import loss
from model import get_generator, get_discriminator, build_generator, build_discriminator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class GAN(object):
    def  __init__(self, alpha_g, alpha_d, trial, version):
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 100
        self.EPOCHS = 250
        self.test_size = 10000
        self.alpha_g = alpha_g
        self.alpha_d = alpha_d
        self.trial = trial
        self.version = version
        self.noise_dim = 28*28
        self.num_examples_to_generate = 16

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])    
        (self.dataset, self.real_mu, self.real_sigma) = data.load_mnist(self.BUFFER_SIZE, self.BATCH_SIZE) #ADD to train function
        self.generator = build_generator()  #Add to build function
        self.discriminator = build_discriminator() #Add to build function
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)

        self.checkpoint_dir = 'data/renyiganV_' + str(self.version) + '/AlphaG=' + str(self.alpha_g)  + '_AlphaD='+ str (self.alpha_d) + '/trial' + str(self.trial) + './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.image_dir = 'data/renyiganV_' + str(self.version) + '/AlphaG=' + str(self.alpha_g)  + '_AlphaD='+ str (self.alpha_d) + '/trial' + str(self.trial) + '/images'
        self.plot_dir = 'data/renyiganV_' + str(self.version) + '/AlphaG=' + str(self.alpha_g)  + '_AlphaD='+ str (self.alpha_d) + '/trial' + str(self.trial)+ '/plots'


        self.make_directory('data')
        self.make_directory('data/renyiganV_' + str(self.version))
        self.make_directory('data/renyiganV_' + str(self.version) + '/AlphaG=' + str(self.alpha_g)  + '_AlphaD='+ str (self.alpha_d))
        self.make_directory('data/renyiganV_' + str(self.version) + '/AlphaG=' + str(self.alpha_g)  + '_AlphaD='+ str (self.alpha_d) + '/trial' + str(self.trial))
        self.make_directory(self.image_dir)
        self.make_directory(self.plot_dir)

        if (version == 1):
            self.generator_loss = loss.generator_loss_renyi
            self.discriminator_loss = loss.discriminator_loss_rgan
        elif (version == 2):
            self.generator_loss = loss.generator_loss_renyiL1
            self.discriminator_loss = loss.discriminator_loss_rgan
        elif (version == 3):
            self.generator_loss = loss.generator_loss_original
            self.discriminator_loss = loss.discriminator_loss_rgan
        elif(version == 4):
            self.generator_loss = loss.generator_loss_rgan
            self.discriminator_loss = loss.discriminator_loss_rgan
        else:
            quit()
        


    
    @staticmethod
    def make_directory(PATH):
        if not os.path.exists(PATH):
            os.mkdir(PATH)

    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_out = self.discriminator(images, training=True)
            fake_out = self.discriminator(generated_images, training=True)

            # disc_loss = loss.discriminator_loss_rgan(real_out,fake_out, self.alpha_d)
            # gen_loss = loss.generator_loss_rgan(fake_out, self.alpha_g)

            disc_loss = self.discriminator_loss(real_out,fake_out, self.alpha_d)
            gen_loss = self.generator_loss(fake_out, self.alpha_g)


        # this is printing all the red numbers and will show 'nan' if broken
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        # tf.print(disc_loss, gen_loss)
        return disc_loss, gen_loss


    def train(self):
        disc_hist, gen_hist, fid_hist = list(), list(), list()
        best_fid = 10000
        best_epoch = 0
        for epoch in range(self.EPOCHS):
            start = time.time()
            dloss = 0
            gloss = 0
            batchnum = 0
            fid = 0
            for image_batch in self.dataset:
                batchnum = batchnum + 1
                (d, g) = self.train_step(image_batch)
                dloss = dloss + d
                gloss = gloss + g
            fid = self.calculate_fid()

            if (fid < best_fid):
                best_fid = fid
                best_epoch = epoch + 1

            fid_hist.append(fid)
            disc_hist.append(dloss/(batchnum))
            gen_hist.append(gloss/batchnum)


            if (epoch + 10) % 10 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print ('Time for epoch {} is {} sec. FID: {}'.format(epoch + 1, time.time()-start, fid))
        
            self.plot_and_save_history(disc_hist, gen_hist, fid_hist, best_fid, best_epoch)
            self.generate_and_save_images(self.generator, epoch + 1, self.seed)


    def calculate_fid(self):
        fake_images = self.generator(tf.random.normal([self.test_size, self.noise_dim])) 
        fake_images = fake_images.numpy()
        fake_images = fake_images.reshape(fake_images.shape[0], 28*28).astype('float32')

        fake_images = (fake_images * 127.5 + 127.5) / 255.0
        fake_mu = fake_images.mean(axis=0)
        fake_sigma = np.cov(np.transpose(fake_images))
        covSqrt = sp.linalg.sqrtm(np.matmul(fake_sigma, self.real_sigma))
        
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        fidScore = np.linalg.norm(self.real_mu - fake_mu) + np.trace(self.real_sigma + fake_sigma - 2 * covSqrt)

        return fidScore


    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)

        plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.image_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()
            # plt.show()


    def plot_and_save_history(self, d_hist, g_hist, fid_hist, best_fid, best_epoch):

        with open(self.plot_dir + '/History.txt', 'w') as output:
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
        plt.savefig(self.plot_dir + '/Loss_History.png')
        plt.close()

        plt.figure(2)

        # plot discriminator accuracy
        plt.plot(fid_hist)
        plt.title("FID History")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.savefig(self.plot_dir + '/FID_Plot.png')
        plt.close()






t = [9, 10]
v = 2
a_g = [3, 9]
a_d = [0.5, 0.1]

for x in t:
    for y in a_g:
        for z in a_d:
            model = GAN(y, z, x, v)
            model.train()