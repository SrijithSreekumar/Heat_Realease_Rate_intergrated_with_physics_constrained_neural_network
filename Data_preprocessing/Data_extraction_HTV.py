# To visualize the normalized heat release rate. 
# A Threshold values can be interpolated such that only high heat release regions are captured 
# These normalized High heat release region would be used for training.

# DINO -  The in house DNS code. The saptial and temporal data are at different Equivalence ratio is simulated.

#import modules
import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
import tensorflow as tf

#Read the file containing information

filename = 'F:\HRR\phi_(X)_dino_yyyy'  # Any randomly selected data
f = h5py.File(filename, 'r')

for key in f.keys():
#Get the HDF5 group
    group = f[key]

#Checkout what keys are inside that group.
for key in group.keys():

    #extrat the heat release data
    heat = np.array(f['hr'])

    # convert the heat to an 1D and visualize the result
    # reshape to 1d
    heat = tf.reshape(heat, shape=[-1, 1])
    print(heat.shape)

    # Do normalization on the 1D and visualize the heat release region
    normHRR = minmax_scale(heat, axis=0)

    # Again reshape the normalized result and visualize in VisIT or  Paraview
    normHRR = tf.reshape(normHRR, shape=[2049, 1024])
    print(normHRR)

    # Saving the output of the normalized heat release to a hdf5 file
    h5f = h5py.File('norm_pred_heat_release_xxx.h5', 'w')
    h5f.create_dataset('normHRR', data=normHRR)
f.close()
