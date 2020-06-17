import tensorflow as tf
import create_models
import define_loss
import numpy as np
import helper
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_DIM = 100
DATA_PATH = './data/gen4up/'
RESULT_PATH = './results/pokemon/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#Load and prepare dataset
#normalize the inputs between -1 and 1
pokemon = helper.load_data(DATA_PATH)

#display a random subset of images, uncomment to double check data looks fine
#helper.display_image_grid(pokemon)
pokemon = (pokemon-127.5)/127.5

#load the dataset into a tf dataset, shuffle, and separate into batches
dataset = tf.data.Dataset.from_tensor_slices(pokemon).shuffle(len(pokemon)).batch(BATCH_SIZE)

#Create Generator
generator = create_models.build_generator()
noise = tf.random.normal((1,INPUT_DIM))
fake_image = generator(noise, training=False)

#shows the fake image, commented out because I know it works
#helper.display_fake_image(fake_image[0])

#Create Discrimator
discriminator = create_models.build_discriminator()
print(discriminator(fake_image),'for a fake image')
print(discriminator(pokemon[0].reshape(1,64,64,3)),'for a real image')

#Define Loss
disc_loss = define_loss.disc_loss
gen_loss = define_loss.gen_loss

#Set up optimizers
#Some contention on LR of 0.0001 or 0.0002. Also default beta of 0.9 or 0.5?
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
#a paper suggests using SGD for the discriminator
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

#create a reference vector to get visual feedback on performance
reference_vectors = tf.random.normal((25,100))

gen_loss_hist = []
disc_loss_hist = []

#a single training step, generator makes BATCH_SIZE images, discriminator decides on 2x sample? (confirm?)
#adapted from the method at the tf tutorials https://www.tensorflow.org/tutorials/generative/dcgan
@tf.function
def train_step(real_ims):
    #Generate noise vectors
    noise = tf.random.normal((BATCH_SIZE,INPUT_DIM))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generate fake images from noise
        gen_ims = generator(noise, training=True)
        #input real images to discriminator
        real_output = discriminator(real_ims, training=True)
        #input real images to discriminator
        fake_output = discriminator(gen_ims,training=True)
        #calculate generator loss
        generator_loss = gen_loss(fake_output)
        #calculate discriminator loss
        discriminator_loss = disc_loss(real_output,fake_output)

    #update weights using gradients
    gen_grads = gen_tape.gradient(generator_loss,generator.trainable_variables)
    disc_grads = disc_tape.gradient(discriminator_loss,discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads,generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads,discriminator.trainable_variables))

    return (generator_loss,discriminator_loss)

def train(dataset, epochs):
    for epoch in range(epochs):
        #set up a timer to get an idea of training per epoch
        start = time.time()

        #in each
        for image_batch in dataset:
            l = train_step(image_batch)
            gen_loss_hist.append(l[0])
            disc_loss_hist.append(l[1])

        
        helper.save_image_grid(generator,epoch+1,reference_vectors,RESULT_PATH)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        
train(dataset,NUM_EPOCHS)

#Analyse Data
#TODO graph losses over time

#https://github.com/soumith/ganhacks has some awesome tips to optimize!