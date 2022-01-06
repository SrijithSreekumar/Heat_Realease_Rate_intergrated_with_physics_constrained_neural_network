########################################################### Recontruction of 

################################ Import the modules required ##################################################

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time as t

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
np.set_printoptions(precision=8)

seed = 0

#Read the starting time
start_time = t.time()
	
############################################# Read the dataset ##############################################
#Read directory
file = 'major_chemical_marker.hdf5'
f = h5py.File(file, 'r')

# Get the feature and the label
# The dataset have larger points HHR region [data= Augumentation]
dataset = np.array(f.get('major_chemical_marker'))
print(dataset.shape)

# Separate the feature and the label
X = dataset[:,:-1]
Y = dataset[:,[-1]]
#print(X,Y.shape)

################################################ Normalize the dataset ###########################################
# Now normalize the train and test data		
scale_X = MinMaxScaler()
scale_Y = MinMaxScaler()

# Normalize the trainig data
X_n = scale_X.fit_transform(X)
Y_n = scale_Y.fit_transform(Y)

print(X_n,Y_n)

#load the model
model = tf.keras.models.load_model('updated_model.h5')

# Get summary
model.summary()

############################################### The recontruction #################################################

# Get the data from a randomly selected species.
doc = '/home/sreekumar/Downloads/dino_res_9200.h5' #Select an random DNS file 
f1 = h5py.File(doc, 'r')

# extract the info of chemical marker and the heat release rate
y_nh = np.array(f1['Y_NH'])
y_nh2 = np.array(f1['Y_NH2'])
y_oh = np.array(f1['Y_OH'])
temp = np.array(f1['temper'])
#  The outout lable
heat = np.array(f1['hr'])

# make the tensor 
tensor = np.array([[y_nh],[y_nh2],[y_oh],[temp],[heat]])
tensor = tf.transpose(tensor, perm=[2, 3, 0, 1])
tensor = tf.reshape(tensor, shape=[-1, 5])

X_reconstruct = tensor[:,:-1]
Y_ori = tensor[:, -1]
Y_ori = tf.reshape(Y_ori , shape=[len(Y_ori),1])

X_test_n = scale_X.transform(X_reconstruct)
Y_test_n = scale_Y.transform(Y_ori)
print(X_test_n, Y_test_n)

pred = model.predict(X_test_n)
print(pred)

# Inverse the predicted output
pred = scale_Y.inverse_transform(pred)

# Reshape the prediction output 
prediction = tf.reshape(pred, shape=[2049, 1024])

# Plot the fitting of the prediction
b = plt.figure(1)
plt.scatter(pred, Y_ori, s=1)
plt.xlabel('ypred')
plt.ylabel('Y_ori')
# predicted values
plt.plot(Y_ori, Y_ori, color='k')
b.savefig('pred_2.png')

# save the reconstructed image
c = h5py.File('reconstructed_ori.h5','w')
c.create_dataset('recon_ori',data=prediction)
c.close()

# compute the elapsed time
elapsed_time = t.time() - start_time
print('elapsed time for the reading script=', round(elapsed_time, 3), 'sec \n')
