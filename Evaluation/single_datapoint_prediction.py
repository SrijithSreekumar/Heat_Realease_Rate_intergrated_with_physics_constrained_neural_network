import numpy as np
import h5py
import tensorflow as tf
np.set_printoptions(precision=8)

#For reproducibility
from numpy.random import seed
seed(0)
tf.random.set_seed(0)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Extract the data
file = 'The\HDF5\data' # From that specific directory
f1 = h5py.File(file, 'r')

# Get the preprocessed_data
X_train = f1.get('X')
Y_train = f1.get('Y')

# Convert the data to tensor
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train)
print(Y_train)
print(X_train.shape, Y_train.shape)

#extract any one data point randomly from the file
rmd  =  100
X_train_1_datapoint = X_train[rmd, :]
#transpose the row to the coloumn vector
X_train_1_datapoint= np.array([X_train_1_datapoint])
X_train_1_datapoint_t = X_train_1_datapoint.T
X_train_1_datapoint_t = tf.transpose(X_train_1_datapoint_t, perm=[1,0])
Y_train_checkpoint = Y_train[rmd]# reference for output
print(X_train_1_datapoint_t, Y_train_checkpoint)
print(X_train_1_datapoint_t.shape, Y_train_checkpoint.shape)

#  Evaluate the model performance using the trained model weights
#  Restore the model with its weights and the optimizer
model = tf.keras.models.load_model('The_trained_model')

prediction = model.predict(X_train_1_datapoint_t)
prediction = np.array(prediction)
print("The predicted result by the model is:",prediction)

#Calculation of the relative
relative_error = ((np.abs(prediction-Y_train_checkpoint))/Y_train_checkpoint)*100
print('The relative error percentage is:',relative_error)

