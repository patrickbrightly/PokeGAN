# PokeGAN
Using a Deep Convolutional Generative Adverserial Network (DCGAN) to generate fake pokemon sprites.

## Dataset
* All sprites were downloaded [this website](https://veekun.com/dex/downloads)
* All images were scaled down to 64x64 RGB, superimposed onto a white background so all alphas were turned white

* Sprites from generation 4 and up were chosen since the art tended to be similar from those games and the newer ones

Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5
------|-------|-------|-------|------
![gen1](images/g1blue.png "Blue Bulbasaur")|![gen2](images/g2gold.png "Gold Bulbasaur")|![gen3](images/g3emerald.png "Emerald Bulbasaur")|![gen4](images/g4diamond.png "Diamond Bulbasaur")|![gen5](images/g5black.png "Black Bulbasaur")
![gen1](images/g1red.png "Red Bulbasaur")|![gen2](images/g2silver.png "Silver Bulbasaur")|![gen3](images/g3ruby.png "Ruby Bulbasaur")|![gen4](images/g4plat.png "Platinum Bulbasaur")|![gen5](images/g5black.png "Black Bulbasaur")
![gen1](images/g1yellow.png "Yellow Bulbasaur")|![gen2](images/g2crystal.png "Crystal Bulbasaur")||![gen4](images/g4soulsilver.png "SSilver Bulbasaur")|

* All Unown sprites were removed due to their similarity to letters and drastic difference aesthetically to other pokemon. What a boring pokemon!

## Network Architectures
* This network used Tensorflow with Keras
* Generator was built with the following layers
    1. Input (1x100) -> Dense -> BatchNormalization -> LeakyReLU -> Reshape(4x4x1024)
    2. Conv2DTranspose -> BatchNormalization -> LeakyReLU
    3. Conv2DTranspose -> BatchNormalization -> LeakyReLU
    4. Conv2DTranspose -> BatchNormalization -> LeakyReLU
    5. Conv2DTranspose -> tanh
    * This outputs a 64x64x3 tensor which can be viewed as an image
* Discriminator was built with the following layers
    1. Input (64x64x3) -> Conv2D -> LeakyReLU -> Dropout
    2. Conv2D -> LeakyReLU -> Dropout
    3. Conv2D -> LeakyReLU -> Dropout
    4. Flatten -> Dense
    * This outputs a score, which should be 1 if the image is real, 0 if fake

## Results
### Epoch 1
![epoch 1](images/epoch_0001.png "Sprites after 1 epoch")
### Epoch 100
![epoch 100](images/epoch_0100.png "Sprites after 100 epochs")
### Epoch 200
![epoch 200](images/epoch_0200.png "Sprites after 200 epochs")
### Epoch 300
![epoch 300](images/epoch_0300.png "Sprites after 300 epochs")

Here are a few of my favourites!

![](images/1002_fake.png)![](images/998_fake.png)![](images/037_fake.png)![](images/326_fake.png)![](images/531_fake.png)![](images/708_fake.png)![](images/067_fake.png)![](images/074_fake.png)![](images/210_fake.png)![](images/351_fake.png)![](images/471_fake.png)![](images/515_fake.png)![](images/584_fake.png)![](images/606_fake.png)![](images/722_fake.png)![](images/883_fake.png)

## Next Steps
* Testing the architectures
    * Adding dropouts to the generator [(here for more details)](https://github.com/soumith/ganhacks)
    * Optimizing the hyperparameters:
        * Optimizers: LR, Beta1
* Putting the images through an upsampling algorithm to get more detail
* Have an artist use them as inspiration