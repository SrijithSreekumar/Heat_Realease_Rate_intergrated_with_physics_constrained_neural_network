                ############################################ Molar fraction of the major species after Auto-diff ############################################

# Import modules 
import numpy as np
import h5py
import os
import time as t
from sklearn.preprocessing import minmax_scale
import tensorflow as tf

# record the time
start_time = t.time()

# ----------------------------------------------
# Adjust the following variables (user-defined)
# ----------------------------------------------
dir_in = 'F:\HRR'
single_file = False  # true for single file, false for multiple file
file_number = 2850  # file tag
grid_type = 1  # 1=>uniform grid or 2=>non-uniform grid
LowMach = True  # True (for combustion), False (for cold flow)
analy_dim = 1  # 0 is full analysis, 1 is for 2d analysis
file_info = True  # display the file information
debug = True  # debugging

# ----------------------------------------------
# constants
# ----------------------------------------------
uniform_grid = 1
nonuniform_grid = 2
counter = 0
TwoDim = 1

# ----------------------------------------------
# create Hdf5 file to write the information
# ----------------------------------------------
ppd = h5py.File("molar_fraction_major_chemical_marker.hdf5", "w")

# ----------------------------------------------
# start reading the file one-by-one to record
# the useful information in one file.
# ----------------------------------------------
i = 0
for root, subdirectories, files in os.walk(dir_in):
    for file in files:
        # Read hdf5 file
        file = os.path.join(root, file)
        f = h5py.File(file, 'r')

        #To view the header of the current hdf5 file.
        # Read the file name with their specific equivalence ratio
        Hdf5_group_header = str(os.path.join(root, file))
        Hdf5_group_header = Hdf5_group_header.replace('F:\\HRR\\phi_', '')
        Hdf5_group_header = Hdf5_group_header.replace('\\dino_res', '')


        y_nh = np.array(f['Y_NH'])      #1
        y_nh2 = np.array(f['Y_NH2'])    #2
        y_oh = np.array(f['Y_OH'])      #3

        # Extract density for conversion
        density = np.array(f['rho'])
        # temperature as additional input
        temp = np.array(f['temper'])#4
        # output attribute[HRR]
        heat = np.array(f['hr'])#21

# #
        # Upending the list and Forming it as tensor, Creating a complied Dataset
        # The major species of interest are being taken
        model_dataset = np.array([[y_nh], [y_nh2], [y_oh], [density], [temp], [heat]])

        #Input and output shape
        model_dataset = tf.transpose(model_dataset, perm=[2, 3, 0, 1])
        #print(model_dataset.shape)
        model_dataset = tf.reshape(model_dataset, shape=[-1, 6])
        #print(model_dataset)

        # allocating Xtrain and Ytrain
        not_normalized = model_dataset[:, :]
        #print(not_normalized.shape)
        Heat_release = model_dataset[:, -1]
        Heat_release = tf.reshape(Heat_release, shape=[-1, 1])
        #print(Heat_release.shape)

        # Removing the redundant data from the previous approach of extracting the high heat release data points and masking
        # that index positions to the  model dataset , which has the input features and the original heat release data.

        Norm_heat_release = minmax_scale(Heat_release, axis=0)

        # Removing the redundant data from the dataset and setting the
        HRR_tHV1 = 0.1 # as informed by the professor
        mask_1 = tf.where(Norm_heat_release > HRR_tHV1)
        redundant_index = mask_1[:, 0]


        # Initially the redundant data is being removed from the dataset
        filtered_model_dataset_points = tf.gather(not_normalized, redundant_index, axis=0)

        print("The current file:", Hdf5_group_header)
        #print(filtered_model_dataset_points.shape)
        print("The Appended shape  file:")

        #Appending all the Not_ normalized data points
        if i == 0:
            ppd_FMDP = ppd.create_dataset('molar_fraction', data=filtered_model_dataset_points, maxshape=(None, 6))
        else:
            ppd_FMDP.resize(ppd_FMDP.shape[0]+filtered_model_dataset_points.shape[0], axis=0)
            ppd_FMDP[-filtered_model_dataset_points.shape[0]:] = filtered_model_dataset_points
        print(ppd_FMDP.shape)
        i += 1
ppd.close()

# compute the elapsed time
elapsed_time = t.time() - start_time
print('elapsed time for the reading script=', round(elapsed_time, 3), 'sec \n')
# --end the program
# sys.exit("program end")
print("program end")
