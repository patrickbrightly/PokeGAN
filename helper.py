from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import random

#this method loads all the photos from a folder into
def load_data(path,num=None):
    if not os.path.exists(path):
            raise Exception("input path is not valid")
    dataset = []
    files = glob.glob(path+'*') 
    for f in files:
        try:
            img = Image.open(f)
            dataset.append(np.asarray(img))
        except UnidentifiedImageError:
            print(f, 'was not added to the dataset. Check filetype')
    return np.array(dataset)

def load_random_subset(path,num):
    dataset=[]
    files = glob.glob(path+'*')
    if num>len(files):
        raise Exception("cannot load more data than dataset size")
    ind = set()
    while(len(ind)<num):
        ind.add(random.randint(0,len(files)-1))
    for x in ind:
        try:
            img = Image.open(files[x])
            dataset.append(np.asarray(img))
        except UnidentifiedImageError:
            print(files[x], 'was not added to the dataset. Check filetype')
    return np.array(dataset)

#takes an array and displays it as a grid of images
def display_image_grid(array_images):
    rand_ims=[]
    nums = []
    for x in range(25):
        num=random.randint(0,len(array_images)-1)
        nums.append(num)
        rand_ims.append(array_images[num])

    x = [Image.fromarray(im) for im in rand_ims]
    plt.subplot(5,5,1,frameon=False)

    for idx in range(len(x)):
        plt.subplot(5,5,idx+1)
        plt.imshow(x[idx])
        plt.axis('off')

    plt.show()

def display_fake_image(img_array):
    img = np.array((img_array*127.5)+127.5)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()

def save_image_grid(model, epoch, input_vector,path):
    gen_vectors = model(input_vector, training=False)
    if not os.path.exists(path):
        os.makedirs(path)
  
    plt.subplot(5,5,1,frameon=False)

    for idx in range(gen_vectors.shape[0]):
        img = np.array((gen_vectors[idx]*127.5)+127.5)
        img = img.astype('uint8')
        img = Image.fromarray(img)
        plt.subplot(5,5,idx+1)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(os.path.join(path,'epoch_{:04d}.png'.format(epoch)))
    #plt.show()
    plt.close()