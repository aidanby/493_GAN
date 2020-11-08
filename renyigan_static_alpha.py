import os
import time
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape
import utils
import numpy as np

alpha_num, trial_number, version_num, seed_num = input("Alpha, trial number, version, seed: ").split()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(int(seed_num))
tf.random.set_random_seed(int(seed_num))


class GAN(object):
    def __init__(self, alpha, trial_num, version):
        self.batch_size = 100
        self.n_classes = 10
        self.buffer_size = 50000
        self.training = True
        self.alpha = alpha
        self.version = version
        self.trial_num = trial_num
        self.noise_dim = 28 * 28
        self.dropout_constant = 0.6
        self.epsilon = 1e-8  # To ensure the log doesn't blow up to -infinity
        self.predictions = []
        self._make_directory('data')
        self._make_directory('data/renyigan' + str(self.alpha))
        self._make_directory('data/renyigan' + str(self.alpha) + '/v' + str(self.version))
        self._make_directory('data/renyigan' + str(self.alpha) + '/v' + str(self.version) + '/trial' + str(self.trial_num))

    @staticmethod
    def _make_directory(PATH):
        if not os.path.exists(PATH):
            os.mkdir(PATH)

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                            train_data.output_shapes)
            img, _ = self.iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])

            self.train_init = self.iterator.make_initializer(train_data)
            self.test_init = self.iterator.make_initializer(test_data)

    def build_generator(self):
        with tf.name_scope('generator') as scope:
            model = Sequential(name=scope)
            model.add(Dense(7 * 7 * 256, use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01), input_shape=(self.noise_dim,)))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

            model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False,
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 28, 28, 1)

            return model

    def build_discriminator(self):
        with tf.name_scope('discriminator') as scope:
            model = Sequential(name=scope)
            model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))

            return model

    # Vanilla DCGAN discriminator loss function
    def dis_loss_vgan(self):
        with tf.name_scope('disLossVGAN'):
            real_loss = -tf.math.log(self.real_output + self.epsilon)
            real_loss = tf.math.reduce_mean(real_loss)

            fake_loss = -tf.math.log(1 - self.fake_output + self.epsilon)
            fake_loss = tf.math.reduce_mean(fake_loss)
            gradients = tf.gradients(-tf.math.log(1 / self.real_output - 1), [self.img])[0]
            r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            dis_loss = real_loss + fake_loss + 5 * r1_penalty
            return dis_loss

    # Vanilla DCGAN discriminator loss function with gradient penalty
    def dis_loss_vgan_gp(self):
        with tf.name_scope('disLossVGANGP'):
            real_loss = -tf.math.log(self.real_output + self.epsilon)
            real_loss = tf.math.reduce_mean(real_loss)

            fake_loss = -tf.math.log(1 - self.fake_output + self.epsilon)
            fake_loss = tf.math.reduce_mean(fake_loss)

            gradients = tf.gradients(-tf.math.log(1 /self.real_output - 1), [self.img])[0]
            r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
            dis_loss = real_loss + fake_loss  + 5 * r1_penalty

            return dis_loss

    # RenyiGAN discriminator loss function
    def dis_loss(self):
        with tf.name_scope('disLoss'):
            real_loss = tf.math.reduce_mean(tf.math.pow(self.real_output, (self.alpha - 1)
                                                        * tf.ones_like(self.real_output)))
            real_loss = 1.0 / (self.alpha - 1) * tf.math.log(real_loss + self.epsilon) + tf.math.log(2.0)
            f = tf.math.reduce_mean(tf.math.pow(1 - self.fake_output,
                                                (self.alpha - 1) * tf.ones_like(self.fake_output)))
            gen_loss = 1.0 / (self.alpha - 1) * tf.math.log(f + self.epsilon) + tf.math.log(2.0)
            dis_loss = - real_loss - gen_loss
            return dis_loss

    # Vanilla DCGAN generator l1 loss function
    def gen_loss_vgan_l1(self):
        with tf.name_scope('genLossVGANL1'):
            fake_loss = tf.math.log(1 - self.fake_output + self.epsilon)
            fake_loss = tf.math.reduce_mean(fake_loss)
            gen_loss = tf.math.abs(fake_loss + tf.math.log(2.0))
            return gen_loss

    # Vanilla DCGAN generator loss function
    def gen_loss_vgan(self):
        with tf.name_scope('genLossVGAN'):
            fake_loss = - tf.math.log(self.fake_output + self.epsilon)
            gen_loss = tf.math.reduce_mean(fake_loss)
            return gen_loss

    # RenyiGAN generator loss function (has l1 norm incorporated)
    def gen_loss_l1(self):
        with tf.name_scope('genLossL1'):
            f = tf.math.reduce_mean(tf.math.pow(1 - self.fake_output,
                                                (self.alpha - 1) * tf.ones_like(self.fake_output)))
            gen_loss = tf.math.abs(1.0 / (self.alpha - 1) * tf.math.log(f + self.epsilon) + tf.math.log(2.0))
            return gen_loss

    # RenyiGAN generator loss function
    def gen_loss(self):
        print("RenyiGAN " + str(self.alpha))
        with tf.name_scope('genLoss'):
            f = tf.math.reduce_mean(tf.math.pow(1 - self.fake_output,
                                                (self.alpha - 1) * tf.ones_like(self.fake_output)))
            gen_loss = 1.0 / (self.alpha - 1) * tf.math.log(f + self.epsilon)
            return gen_loss

    def optimize(self):
        self.gen_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, name="generator_optimizer")
        self.gen_opt_minimize = self.gen_opt.minimize(self.gen_loss_value, var_list=self.generator.trainable_variables)
        self.dis_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, name="discriminator_optimizer")
        self.dis_opt_minimize = self.dis_opt.minimize(self.dis_loss_value,
                                                      var_list=self.discriminator.trainable_variables)

    def build(self):
        self.get_data()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.fake_output_images = self.generator(tf.random.normal([self.batch_size, self.noise_dim]))
        self.fake_output = self.discriminator(self.fake_output_images)
        self.real_output = self.discriminator(self.img)
        if self.alpha != 1.0:
            if self.version == 1 or self.version == 3:
                print("RenyiGAN no L1 normalization")
                self.gen_loss_value = self.gen_loss()
            else:
                print("RenyiGAN with L1 normalization")
                self.gen_loss_value = self.gen_loss_l1()
        else:
            if self.version == 1 or self.version == 3:
                print("Vanilla GAN No L1 normalization")
                self.gen_loss_value = self.gen_loss_vgan()
            else:
                print("Vanilla GAN with L1 normalization")
                self.gen_loss_value = self.gen_loss_vgan_l1()
        if self.version == 1 or self.version == 2:
            self.dis_loss_value = self.dis_loss_vgan()
        else:
            self.dis_loss_value = self.dis_loss_vgan_gp()
        self.optimize()

    def train_one_epoch(self, sess, init, epoch):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss_gen = 0
        total_loss_dis = 0
        n_batches = 0
        try:
            while True:
                _, disLoss = sess.run([self.dis_opt_minimize, self.dis_loss_value])
                _, genLoss = sess.run([self.gen_opt_minimize, self.gen_loss_value])
                total_loss_gen += genLoss
                total_loss_dis += disLoss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        self.save_generated_images(sess, epoch)
        print('Average generator loss at epoch {0}: {1}'.format(epoch, total_loss_gen / n_batches))
        print('Average discriminator loss at epoch {0}: {1}'.format(epoch, total_loss_dis / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def save_generated_images(self, sess, epoch):
        temp = self.generator(tf.random.normal([self.buffer_size, self.noise_dim]))
        temp = sess.run(temp)
        if len(self.predictions) > 0:
            self.predictions.pop(0)
        self.predictions.append(temp)
        self._make_directory('data/renyigan' + str(self.alpha) 
                + 'v' + str(self.version) + '/trial' + str(self.trial_num) + '/alpha' + str(self.alpha))
        np.save('data/renyigan' + str(self.alpha) + 'v' + str(self.version) + '/trial' + str(self.trial_num) + '/alpha'
                + str(self.alpha) + '/predictions' + str(epoch), self.predictions)

    def train(self, n_epochs):
        self._make_directory('checkpoints')
        self._make_directory('checkpoints/renyigan' + str(self.alpha))
        self._make_directory('checkpoints/renyigan' + str(self.alpha) + '/v' + str(self.version))
        self.cpt_PATH = 'checkpoints/renyigan' + str(self.alpha) + '/v' + str(self.version) + '/trial' + str(self.trial_num)
        if self.trial_num == 1:
            self._make_directory(self.cpt_PATH)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.train_init)
            checkpoint = tf.train.Saver(
                {'generator_optimizer': self.gen_opt, 'discriminator_optimizer': self.dis_opt,
                 'generator': self.generator, 'discriminator': self.discriminator, 'iterator': self.iterator},
                max_to_keep=3)
            for epoch in range(n_epochs):
                if self.trial_num == 1:
                    if epoch % 10 == 0:
                        save_path = checkpoint.save(sess, self.cpt_PATH + str('/ckpt'), global_step=epoch)
                        print("Saved checkpoint for step {}: {}".format(int(epoch), save_path))
                print("Alpha value: " + str(self.alpha))
                self.train_one_epoch(sess, self.train_init, epoch)


model = GAN(round(float(alpha_num), 1), int(trial_number), int(version_num))
model.build()
model.train(n_epochs=250)
