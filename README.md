# image-classification-using-cnn
STEPS:
1.INSTALL PYTHON:
              Link to install python:click here to redirect to python download page is https://www.python.org/downloads/
       note:ignore id python is already installed
2.INSTALL ANACONDA:
              Link to install python:Click here to redirect to anaconda download page is https://www.anaconda.com/distribution/#windows

               After successful installation of anaconda prompt launch jupyter notebook           

3.CODING PART:
       Go through this blog(click here) to understand CNN algorithm.
         Open your jupyter notebook
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

In line 1, we’ve imported Sequential from keras.models, to initialise our neural network model as a sequential network

In line 2, we’ve imported Conv2D from keras.layers, this is to perform the convolution operation i.e the first step of a CNN, on the training images

In line 3, we’ve imported MaxPooling2D from keras.layers, which is used for pooling operation, that is the step — 2 in the process of building a cnn

In line 4, we've imported Flatten from keras.layers,which is used for Flattening

And finally in line 5, we’ve imported Dense from keras.layers, which is used to perform the full connection of the neural network, which is the step 4 in the process of building a CNN.

Now, we will create an object of the sequential class below:
classifier = Sequential()

Let us now code the Convolution step:
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

From above :
Optimizer parameter is to choose the stochastic gradient descent algorithm.
Loss parameter is to choose the loss function.
Finally, the metrics parameter is to choose the performance metric.

Now we have to train our data:
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

You can find the explanation of what each of the above parameters do here

Instead of training set in line 7 you to paste the path of your image data set folder & in place of test data set paste the path of your testing data.

Fitting data to our model:
classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)

For getting better accuracy steps per epochs must be greater than 5000(takes lot of time to train ).It is nothing but the no of images in train data set.

Epochs are nothing but no.of times you want to repeat the training.


After successful completion of your training its time to validate new images:

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('path of the image', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
prediction = 'yes'
else:
prediction = 'no'
