import tensorflow as tf
from tensorflow.keras import layers

def build_generator(input_size=(100,)):
    #Create the actual model as a sequential model
    model = tf.keras.Sequential(name='generator')
    #Create the first layer with input 100, output the size of the next layer after reshaping
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(input_size)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #reshape the input size to a different volume, must match the number of weights in the dense layer
    model.add(layers.Reshape((4,4,1024)))
    #First Conv2DTranspose Layer, output will be shaped (4*2,4*2,512)
    model.add(layers.Conv2DTranspose(512,(5,5),(2,2),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #Second Conv2DTranspose Layer, output will be shaped (16,16,256)
    model.add(layers.Conv2DTranspose(256,(5,5),(2,2),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #Third Conv2DTranspose Layer, output will be shaped (32,32,128)
    model.add(layers.Conv2DTranspose(128,(5,5),(2,2),padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #Final Conv2DTranspose Layer, output will be shaped (64,64,3)
    model.add(layers.Conv2DTranspose(3,(5,5),(2,2),padding='same',use_bias=False, activation='tanh'))
    model.summary()
    return model

def build_discriminator():
    #Create the actual model as a sequential model
    model = tf.keras.Sequential(name='discriminator')
    #convolve-dropout-activate
    model.add(layers.Conv2D(64,(4,4),(2,2),padding='same',use_bias=False,input_shape=(64,64,3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    #convolve-dropout-activate
    model.add(layers.Conv2D(128,(4,4),(2,2),padding='same',use_bias=False,input_shape=(64,64,3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    #convolve-dropout-activate
    model.add(layers.Conv2D(256,(4,4),(2,2),padding='same',use_bias=False,input_shape=(64,64,3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    #flatten
    model.add(layers.Flatten())
    #output
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model