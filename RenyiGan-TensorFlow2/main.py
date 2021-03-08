# Builds and train the DCGAN model  
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import data
import loss
from model import get_generator, get_discriminator, build_generator, build_discriminator



# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")



BUFFER_SIZE = 60000
BATCH_SIZE = 1
EPOCHS = 50
alpha_g = 2
alpha_d = 0.1
#Change noise_dim to = 28*28 to use Himesh' gen/disc models
noise_dim = 28*28
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

dataset = data.load_mnist(BUFFER_SIZE, BATCH_SIZE)
generator = build_generator()
tf.keras.utils.plot_model(generator, to_file="Generator.png", show_shapes=True)
discriminator = build_discriminator()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-7)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)




@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated_images, training=True)

        gen_loss = loss.generator_loss_original(fake_out)
#         disc_loss = loss.discriminator_loss_original(real_out,fake_out)
        disc_loss1 = loss.discriminator_loss_renyi1(real_out, alpha = 0.1)
        disc_loss2 = loss.discriminator_loss_renyi0(fake_out, alpha = 0.1)
        disc_loss = disc_loss2 + disc_loss1
        #gen_loss = loss.generator_loss_renyiL1(fake_out, alpha_g)
        #disc_loss = loss.discriminator_loss_renyi(real_out,fake_out, alpha_d)
        # this is printing all the red numbers and will show 'nan' if broken
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    tf.print(disc_loss, gen_loss, real_out, fake_out)
    


def train(dataset, epochs):
    for epoch in range(epochs):
        # This will train gen on even epochs and disc on odd ones but doesn't seem to work atm
        # if epoch_count % 2 == 0:
            #     for layers in discriminator.layers:
            #         print('discriminiator layers NOT trainable')
            #         layers.trainable = False
            #     for layers in generator.layers:
            #         print('generator layers trainable')
            #         layers.trainable = True
            # if epoch_count % 2 != 0:
            #     for layers in discriminator.layers:
            #         print('discriminiator layers trainable')
            #         layers.trainable = True
            #     for layers in generator.layers:
            #         print('generator layers NOT trainable')
            #         layers.trainable = False
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
             checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


train(dataset, EPOCHS)





