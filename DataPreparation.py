import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

trainSet = pd.read_csv('train.csv')
testSet = pd.read_csv('test.csv')

# missing values
missingTrain = trainSet.isnull().sum()
missingTrain = missingTrain[missingTrain>0]
missingTrain

missingTest = testSet.isnull().sum()
missingTest = missingTest[missingTest>0]
missingTrain

# no missing values neither in test set or train set

# creating images with the dimension of 28*28*1 from each row in the dataset
X_train_flat = trainSet.drop(columns=['label'])
y_train = trainSet['label']
X_train = X_train_flat.values.reshape(-1,28,28,1)
X_test = testSet.values.reshape(-1,28,28,1)

# creating the label of the images with the one-hot encoder
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_train_cat

# creating train and val sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train_cat, test_size = 0.15)

# showing one image from the train set
plt.imshow(X_train[4][:,:,0])

# image augmentations
from keras_preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=15,
      zoom_range=0.2,
      width_shift_range=0.2,
      height_shift_range=0.3,
      shear_range=0.3,
      #fill_mode='nearest'
      )
validation_datagen = ImageDataGenerator(rescale = 1./255)

training_datagen.fit(X_train)
validation_datagen.fit(X_valid)

