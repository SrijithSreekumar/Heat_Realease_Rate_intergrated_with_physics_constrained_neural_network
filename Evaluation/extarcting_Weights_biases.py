import numpy as np
import tensorflow as tf
import pandas as pd

# Read the file, the pretrained model
file = 'The/HDF5/dataset'
model = tf.keras.models.load_model(file)

# Get the summary
model.summary()
for layer in model.layers:
    print(layer.get_config(), layer.get_weights())

# Read all the weights and biases

# The dropout layer doesn't have the weights and biases

#Input layer
first_layer_weights = model.layers[0].get_weights()[0]
first_layer_biases = model.layers[0].get_weights()[1]
print(first_layer_weights.shape, first_layer_biases.shape)

# Hidden layer 1
second_layer_weights = model.layers[2].get_weights()[0]
second_layer_biases = model.layers[2].get_weights()[1]
print(second_layer_weights.shape, second_layer_biases.shape)

# Hidden layer 2
third_layer_weights = model.layers[4].get_weights()[0]
third_layer_biases = model.layers[4].get_weights()[1]
print(third_layer_weights.shape, third_layer_biases.shape)

# output layer
fourth_layer_weights = model.layers[6].get_weights()[0]
fourth_layer_biases = model.layers[6].get_weights()[1]
print(fourth_layer_weights.shape, fourth_layer_biases.shape)

# create a list
with open("network_X.txt", "wb") as myfile:
    np.savetxt(myfile, first_layer_weights, fmt="%.5f", delimiter=',', newline='\r\n')

with open("network_3.txt", "ab") as myfile:
    np.savetxt(myfile, first_layer_biases, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, second_layer_weights, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, second_layer_biases, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, third_layer_weights, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, third_layer_biases, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, fourth_layer_weights, fmt="%.5f", delimiter=' ', newline='\r\n')
    np.savetxt(myfile, fourth_layer_biases, fmt="%.5f", delimiter=' ', newline='\r\n')

#convert the obtained data set to csv
df = pd.read_fwf('logging.txt')
df.to_csv('logging.csv', mode='w', index=False)
