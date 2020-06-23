import tensorflow as tf
import create_models
import define_loss
import numpy as np
import helper
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

BATCH_SIZE = 64
NUM_EPOCHS = 300
INPUT_DIM = 100
DATA_PATH = './data/gen4up/'
RESULT_PATH = './results/pokemon/'
CHECKPOINT_DIR = './checkpoint/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


if not os.path.exists(RESULT_PATH):
            print(RESULT_PATH,'did not exist, new directory created')
            os.makedirs(RESULT_PATH)

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
#Beta of 0.5 gives more visually convincing results than the default of 0.9
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.7)
#a paper suggests using SGD for the discriminator
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.7)

checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer, discriminator_optimizer=disc_optimizer,
                                 generator=generator, discriminator=discriminator)

#create a reference vector to get visual feedback on performance
reference_vectors = tf.random.normal((100,100))

gen_loss_hist = []
disc_loss_hist = []

#a single training step, generator makes BATCH_SIZE images, discriminator decides on 2x sample? (confirm?)
#adapted from the method at the tf tutorials https://www.tensorflow.org/tutorials/generative/dcgan
#takes a set of real 
@tf.function
def train_step(real_ims):
    """trains the generator and discriminator for one batch of real images

    Parameters
    ----------
    real_ims : tensor, likely an EagerTensor
        The tensor containing the batch of real images to train the discriminator on

    Returns
    -------
    tuple
        a tuple containing the generator loss and the discriminator loss for the training batch
    """

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

        #train each batch
        for image_batch in dataset:
            l = train_step(image_batch)
            gen_loss_hist.append(l[0])
            disc_loss_hist.append(l[1])

        # Save the model every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint.save(file_prefix = os.path.join(CHECKPOINT_DIR,'checkpoint'))
            
        #saves the images generated using the reference vectors created
        helper.save_image_grid(generator,epoch+1,reference_vectors,RESULT_PATH)

        print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time()-start))
        print ('Losses:\t Generator:  --  {:.4f} \t Discriminator: --  {:.4f} '.format(gen_loss_hist[-1],disc_loss_hist[-1]))

        
train(dataset,NUM_EPOCHS)

#Analyse Data
helper.graph_losses(gen_loss_hist,disc_loss_hist,RESULT_PATH)

#https://github.com/soumith/ganhacks has some awesome tips to optimize!