#Load libraries

from keras.datasets import mnist

from keras.layers import *
from keras.layers import LeakyReLU 
from keras.models import Sequential, Model 
from tensorflow.keras.optimizers import Adam
import keras 

import numpy as np
import matplotlib.pyplot as plt
---------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore")
------------------------------------------
(X_train , _), (_ , _)= mnist.load_data()
-----------------------------------------
X_train.shape
---------------------
X_train = (X_train-127.5)/127.5

print(X_train.min())
print(X_train.max())
----------------------------
TOTAL_EPOCHS = 50
BATCH_SIZE = 256
HALF_BATCH = 128

NO_OF_BATCHES = int(X_train.shape[0]/BATCH_SIZE)

NOISE_DIM = 100

optimizers = Adam(learning_rate = 0.0002, beta_1 = 0.5)
---------------------------------------------------------------
#Generator Model: Upsampling

generator = Sequential()
generator.add(Dense(units= 7*7*128, input_shape = (NOISE_DIM,)))
generator.add(Reshape((7,7,128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

#(7,7,128) -> (14,14,64)

generator.add(Conv2DTranspose(64, (3,3), strides= (2,2), padding= 'same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

#(14,14,64) --> (28,28,1)

generator.add(Conv2DTranspose(1, (3,3), strides= (2,2), padding= 'same', activation= 'tanh'))


generator.compile(loss = keras.losses.binary_crossentropy , optimizer=optimizers )
generator.summary()
----------------------------------------------------------------------------
#Discriminator Model - Downsampling

#(28,28,1) -> (14,14,64)

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size= (3,3), strides = (2,2), padding= 'same', input_shape= (28,28,1)))

discriminator.add(LeakyReLU(0.2))

#(14,14,64) --> (7,7,128)

discriminator.add(Conv2D(128, kernel_size= (3,3), strides = (2,2), padding= 'same'))

discriminator.add(LeakyReLU(0.2))

#(7,7,128) ---> 6272

discriminator.add(Flatten())
discriminator.add( Dense(100) )
discriminator.add(LeakyReLU(0.2))

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss = keras.losses.binary_crossentropy, optimizer=optimizers)

discriminator.summary()
---------------------------------------------------------------------
from keras.backend import binary_crossentropy
##Combined Model

discriminator.trainable = False

gan_input = Input(shape = (NOISE_DIM), )

generated_img = generator(gan_input )

gan_output = discriminator(generated_img)

#Functional API

model = Model(gan_input , gan_output)

model.compile(loss = keras.losses.binary_crossentropy, optimizer=optimizers)
-----------------------------------------------------------------------------------------
model.summary()
------------------------
X_train = X_train.reshape(-1,28,28,1)
--------------------------------------
X_train.shape
--------------------------------------
def display_images(samples = 25):

  noise = np.random.normal(0,1,size=(samples, NOISE_DIM))

  generated_img = generator.predict(noise)

  plt.figure(figsize = (10,10))
  for i in range(samples):
    plt.subplot(5,5,i+1)
    plt.imshow(generated_img[i].reshape(28,28), cmap = "binary")
    plt.axis('off')

  plt.show()  
  ---------------------------------------------------------------------------------------
  ##Training loop

d_losses = []
g_losses = []

for epoch in range(TOTAL_EPOCHS):

  epoch_d_loss = 0.0
  epoch_g_loss = 0.0

  #Mini batch gradient decent

  for step in range(NO_OF_BATCHES):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Step1 Train Discriminator
    discriminator.trainable = True

    # get the real data

    idx = np.random.randint(0,60000, HALF_BATCH)
    real_imgs = X_train[idx]

    #get the fake images/data
    noise = np.random.normal(0,1, size=(HALF_BATCH,NOISE_DIM))
    fake_imgs= generator.predict(noise)

    # Labels
    real_y = np.ones((HALF_BATCH,1))*0.9
    fake_y = np.zeros((HALF_BATCH,1))

    #now, train D

    d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)

    d_loss = 0.5*d_loss_real + 0.5*d_loss_fake

    epoch_d_loss == d_losses

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Step 2 Train generator

    discriminator.trainable = False 

    noise = np.random.normal(0,1,size=(BATCH_SIZE, NOISE_DIM))

    ground_truth_y = np.ones((BATCH_SIZE,1))

    epoch_g_loss == g_losses

    #+++++++++++++++++++++++++++++++++++++++++++

    print(f"Epoch{epoch+1}, Disc loss { epoch_d_loss/ NO_OF_BATCHES }, Generator loss { epoch_g_loss/ NO_OF_BATCHES }")

    d_losses.append(epoch_d_loss/ NO_OF_BATCHES)
    g_losses.append(epoch_g_loss/ NO_OF_BATCHES)


    if (epoch+1) % 10 == 0:
      generator.save("generator.h5")
      display_images()
