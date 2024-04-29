""" 
Skin Microbiome Autofluorescence Project
Author: Katie Sosnowski 

This code is for training multiple CNN models at a time to determine whether hue image stacks represent healthy (CoNS-only) or dysbiotic (S. aureus-overgrown) bacteria samples.

See https://www.tensorflow.org/tutorials/images/cnn for Python Keras TensorFlow CNN tutorial """

# Machine learning libraries
import tensorflow as tf
from tensorflow import keras as ks
#import tensorflow.keras.legacy.serialization
from tensorflow.keras.optimizers import schedules, Adam, SGD
from tensorflow.keras import datasets, layers, models, regularizers, metrics
# Utility libraries
import sys
import os 
import glob
# Numpy and math
import numpy as np
import math
# Matplotlib for visualization
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot
#skimage
from skimage import data
from skimage.color import rgb2hsv
#pandas
import pandas as pd

#Create learning rate schedulers
#lr_schedule1 = schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
#lr_schedule1 = schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps=10000, end_learning_rate=1e-4, power=4)
#lr_schedule2 = schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps=10000, end_learning_rate=1e-4, power=7)
#lr_schedule3 = schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps = 10000, end_learning_rate=1e-4, power=7)
lr_schedule1 = schedules.InverseTimeDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
lr_schedule2 = schedules.InverseTimeDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
lr_schedule3 = schedules.InverseTimeDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
#opt=Adam(learning_rate=lr_schedule)

#Load train dataset
with np.load('val_e_train5_all/train_labels.npz') as train_labels:
	train_labels = train_labels['arr_0.npy']
with np.load('val_e_train5_all/train_data.npz') as train_data:
	train_data = train_data['arr_0.npy']

#Load test dataset
with np.load('val_e_test5_all/test_labels.npz') as test_labels:
	test_labels = test_labels['arr_0.npy']
with np.load('val_e_test5_all/test_data.npz') as test_data:
	test_data = test_data['arr_0.npy']



"""
------------------------------------FIRST MODEL-----------------------------------------------------
"""

#Create a CNN model with alternating CNN and maxpool layers
model1 = models.Sequential()
#Convolution layer
#model1.add(layers.Conv2D(20, (3,3), padding = 'same', data_format = 'channels_last', activation='relu', activity_regularizer = regularizers.L1(0.01),  input_shape=(None, None, 9)))
#Regularizer
model1.add(layers.Conv2D(25, (3,3), padding = 'same', data_format = 'channels_last', activation='relu', input_shape=(None, None, 9)))
#Max Pooling layer
model1.add(layers.MaxPool2D(pool_size=(5,5), padding='same'))
#Dropout layer
#model1.add(layers.Dropout(0.1)) 
#Convolution layer
model1.add(layers.Conv2D(16, (3,3), padding = 'same', data_format = 'channels_last', activation='relu', input_shape=(None, None, 9)))
#Max Pooling layer
model1.add(layers.MaxPool2D(pool_size=(5,5), padding='same'))
#Dropout layer
#model1.add(layers.Dropout(0.05))
#Convolution layer
model1.add(layers.Conv2D(9, (3,3), padding = 'same', data_format = 'channels_last', activation='relu', input_shape=(None, None, 9)))
#Max Pooling Layer
model1.add(layers.MaxPool2D(pool_size=(5,5), padding='same'))
#Dropout layer
#model1.add(layers.Dropout(0.05))
#Add a flattening and dense layer for output
model1.add(layers.GlobalMaxPool2D()) #this is how you flatten if you have variable-sized inputs
model1.add(layers.Dense(1, activation='sigmoid'))
model1.add(layers.Flatten())

#View a model summary
model1.summary()

#Compile and train the model
#from_logits should be false if already using sigmoid/softmax above
model1.compile(optimizer=Adam(learning_rate=lr_schedule1),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy', metrics.FalseNegatives(), metrics.FalsePositives()])

#Fit the model 
history1 = model1.fit(train_data, train_labels, batch_size = 15, epochs= 100, verbose=1, validation_data=(test_data, test_labels))

#Save the accuracy on the test data
# convert the history.history dict to a pandas DataFrame:     
hist_df1 = pd.DataFrame(history1.history) 

#save to csv: 
hist_csv_file1 = 'human_subjects_Eall_thresholding_test_5.csv'
with open(hist_csv_file1, mode='w') as f:
    hist_df1.to_csv(f)

#Save the model so we can make predictions on validation data later
model1.save('e_all_thresholding_model_5.h5')