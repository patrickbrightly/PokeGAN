import tensorflow as tf
import create_models
import numpy as np
import helper
from PIL import Image

CHECKPOINT_DIR = './checkpoint/'
RESULTS_DIR='./results/gen4up/ind/'

generator = create_models.build_generator()
discriminator = create_models.build_discriminator()
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.7)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.7)


checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer, discriminator_optimizer=disc_optimizer,
                                 generator=generator, discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

def generate_images():
    noise_vectors = tf.random.normal((100,100))
    gen_ims = generator(noise_vectors,training=False)
    gen_ims = np.asarray((gen_ims*127.5)+127.5).astype('uint8')
    return gen_ims

def generate_and_discriminate(cutoff=0.5):
    images = []
    while len(images)<1000:
        noise = tf.random.normal((2000,100))
        fake = generator(noise,training=False)
        predictions = discriminator(fake,training=False)
        for x in range(len(predictions)):
            if predictions[x]>cutoff:
                images.append(np.asarray((fake[x]*127.5)+127.5).astype('uint8'))
        print(len(images),'/1000')
    return np.array(images)

images = generate_and_discriminate(0.9)

helper.display_image_grid(images)
helper.save_individual_images(images,RESULTS_DIR)