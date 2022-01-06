import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import re
import os
import time as t
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout, GaussianNoise
np.set_printoptions(precision=8)

seed = 0

#Read the starting time
start_time = t.time()

#Read directory
file = '/home/sreekumar/Downloads/NHTV_0_1_dataset/Correct_preprocessed_data.hdf5'
f = h5py.File(file, 'r')

# Get the preprocessed_data
FMDP = f.get('FMDP')

# Get the feature and label
X = FMDP[:,:-1]
Y = FMDP[:,-1]
Y = Y.reshape([len(Y),1])
#print(X.shape, Y.shape)

#The additional species
files = '/home/sreekumar/Downloads/NHTV_0_1_dataset/pp_data.hdf5'
f1 = h5py.File(files, 'r')
FMDP_add = f1.get('FMDP_add')
#print(FMDP_add.shape)

#Concatinate the intially driven input with other new set of input
X = np.c_[FMDP_add, X]
print('The complete shape of the input:', X.shape)

# Change the datatype
X = np.float32(X)
Y = np.float32(Y)

#shuffle the dataset
X,Y = shuffle(X, Y, random_state=seed)
print(X, Y)

#test_train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Now normalize the train and test data
scale_X = MinMaxScaler()
scale_Y = MinMaxScaler()

# Normalize the trainig data
X_train_n = scale_X.fit_transform(X_train)
Y_train_n = scale_Y.fit_transform(Y_train)


# Using sequential API with dropout on in hidden layers
model = keras.Sequential(name="Heat_release_rate")
model.add(layers.Dense(40, input_shape=(32,), name="input_layer", activation = 'tanh', kernel_initializer=tf.initializers.GlorotUniform()))
model.add(layers.Dense(24, name="h_layer1", activation = 'tanh'))
model.add(layers.Dense(24,name="h_layer2", activation = 'tanh'))
model.add(layers.Dense(1, activation = "sigmoid", name="output_layers"))

# Get summary
model.summary()

# Model fit
model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.3,momentum=0.8),loss=tf.keras.losses.MeanSquaredError())

# Splitting 20% of dataset for cross validation
history = model.fit(X_train_n, Y_train_n, batch_size=1024, epochs=250, verbose=1, validation_split=0.2, shuffle=True)

#plot
"Loss"
a = plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Saving the plot
a.savefig('loss_plot_250.png',dpi=300)

# Saving the model
model.save('model_newspc_250.h5')

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history)

# save to csv: 
hist_csv_file = 'history_250.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

print("Now the prediction will be carried out.")

#Now normalize the testing data
X_test_n = scale_X.transform(X_test)
Y_test_n = scale_Y.transform(Y_test)

# Now predict the model behaviour
ypred = model.predict(X_test_n)
print(ypred)
r2 = r2_score(Y_test_n, ypred)
rmse = mean_squared_error(Y_test_n, ypred)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# Plot the fitting of the prediction
b = plt.figure(2)
plt.scatter(ypred, Y_test_n, s=5)
plt.xlabel('ypred')
plt.ylabel('Y_test_n')
# predicted values
plt.plot(Y_test_n, Y_test_n, color='k')
b.savefig('prediction250.png')

# compute the elapsed time
elapsed_time = t.time() - start_time
print('elapsed time for the reading script=', round(elapsed_time, 3), 'sec \n')

