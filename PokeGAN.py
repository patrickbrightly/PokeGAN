import tensorflow as tf
import create_models
import define_loss
import numpy as np
import helper
import matplotlib.pyplot as plt
from PIL import Image

BATCH_SIZE = 32
NUM_EPOCHS = 50


#Load and prepare dataset
#normalize the inputs between -1 and 1
pokemon = helper.load_data('./data/gen4up/')
helper.display_image_grid(pokemon)
pokemon = (pokemon-127.5)/127.5

#Create Generator
generator = create_models.build_generator()
noise = tf.random.normal((1,100))
fake_image = generator(noise, training=False)
helper.display_fake_image(fake_image[0])

#Create Discrimator
discriminator = create_models.build_discriminator()
print(discriminator(fake_image),'for a fake image')
print(discriminator(pokemon[0].reshape(1,64,64,3)),'for a real image')

#Define Loss
disc_loss = define_loss.disc_loss
gen_loss = define_loss.gen_loss

#Set up optimizers
#Some contention on LR of 0.0001 or 0.0002. Also default beta of 0.9 or 0.5?
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
#a paper suggests using SGD for the discriminator
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
#train
    #Generate noise vectors
    #generate fake images from noise
    #input fake images to discriminator
    #input real images to discriminator
    #calculate generator loss
    #calculate discriminator loss
    #update weights using gradients

#Analyse Data
#https://github.com/soumith/ganhacks has some awesome tips to optimize!