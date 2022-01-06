#####################################################    PRE-PROCESSING DATASET #############################################################

# Impor the required modules 
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
ppd = h5py.File("new_preprocessed_data.hdf5", "w")

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

        # To view the header of the current hdf5 file.
        # Read the file name with their specific equivalance ratio
        Hdf5_group_header = str(os.path.join(root, file))
        Hdf5_group_header = Hdf5_group_header.replace('F:\\HRR\\phi_', '')
        Hdf5_group_header = Hdf5_group_header.replace('\\dino_res', '')

        # Read the mass fraction
        y_h2o = np.array(f['Y_H2O']) 
        y_h2nn = np.array(f['Y_H2NN'])
        y_hno2 = np.array(f['Y_HNO2'])
        y_hnoh = np.array(f['Y_HNOH'])
        y_hon = np.array(f['Y_HON'])
        y_hono = np.array(f['Y_HONO'])
        y_hono2 = np.array(f['Y_HONO2'])
        y_n2h2 = np.array(f['Y_N2H2'])
        y_n2h3 = np.array(f['Y_N2H3'])
        y_n2h4 = np.array(f['Y_N2H4'])
        y_nh2oh = np.array(f['Y_NH2OH'])
        y_no3 = np.array(f['Y_NO3'])
        y_o3 = np.array(f['Y_O3'])
        y_h = np.array(f['Y_H'])        
        y_h2 = np.array(f['Y_H2'])      
        y_h2no = np.array(f['Y_H2NO'])  
        y_ho2 = np.array(f['Y_HO2'])    
        y_h2o2 = np.array(f['Y_H2O2'])  
        y_hno = np.array(f['Y_HNO'])      
        y_n = np.array(f['Y_N'])        
        y_n2 = np.array(f['Y_N2'])      
        y_n2o = np.array(f['Y_N2O'])    
        y_nh = np.array(f['Y_NH'])      
        y_nh2 = np.array(f['Y_NH2'])    
        y_nh3 = np.array(f['Y_NH3'])    
        y_nnh = np.array(f['Y_NNH'])    
        y_no = np.array(f['Y_NO'])      
        y_no2 = np.array(f['Y_NO2'])    
        y_o = np.array(f['Y_O'])        
        y_o2 = np.array(f['Y_O2'])      
        y_oh = np.array(f['Y_OH'])      
        # temperature as additional input
        temp = np.array(f['temper'])
        # output attribute[HRR]
        heat = np.array(f['hr'])

        # The H*/O* is additional input [Dependent variable]. It is actually not the need for this particular task
        H_fraction = np.array(
            ((1 / 1.0079) * y_h) + ((2 / 2.0158) * y_h2) + ((2 / 32.0219) * y_h2no) + ((2 / 18.0152) * y_h2o)
            + ((1 / 33.0067) * y_ho2) + ((2 / 34.0146) * y_h2o2) + ((1 / 31.014) * y_hno) + ((1 / 15.0146) * y_nh) +
            ((2 / 16.0225) * y_nh2) + ((3 / 17.0304) * y_nh3) + ((1 / 29.0213) * y_nnh) + ((1 / 17.0073) * y_oh))
        O_fraction = np.array(
            ((1 / 32.0219) * y_h2no) + ((1 / 18.0152) * y_h2o) + ((2 / 33.0067) * y_ho2) + ((2 / 34.0146) * y_h2o2) +
            ((1 / 31.014) * y_hno) + ((1 / 44.0128) * y_n2o) + ((1 / 30.0016) * y_no) + ((2 / 46.0055) * y_no2) +
            ((1 / 15.9994) * y_o) + ((2 / 31.9988) * y_o2) + ((1 / 17.0073) * y_oh))
        H_O_ratio = H_fraction / O_fraction

        # Upending the list and Forming it as tensor, Creating a complied Dataset
        model_dataset = np.array([ [y_h2o], [y_h2nn], [y_hno2], [y_hnoh], [y_hon], [y_hono], [y_hono2], [y_n2h2], [y_n2h3], [y_n2h4], [y_nh2oh], [y_no3], [y_o3],
                                  [y_h], [y_h2], [y_h2no], [y_h2o2], [y_hno], [y_ho2], [y_n],[y_n2], [y_n2o], [y_nh], [y_nh2],
                                  [y_nh3], [y_nnh], [y_no], [y_no2], [y_o], [y_o2], [y_oh], [temp], [heat]])

        
        #Input and output shape
        model_dataset = tf.transpose(model_dataset, perm=[2, 3, 0, 1])
        model_dataset = tf.reshape(model_dataset, shape=[-1, 32])
        
        # Allocating Xtrain and Ytrain
        not_normalized = model_dataset[:, :]
        Heat_release = model_dataset[:, -1]
        Heat_release = tf.reshape(Heat_release, shape=[-1, 1])
        #print(Heat_release.shape)

        # Removing the redundant data from the previous approach of extracting the high heat release data points and masking
        # that index positions to the  model dataset , which has the input features and the original heat release data.

        Norm_heat_release = minmax_scale(Heat_release, axis=0)

        # Removing the redundant data from the dataset and setting the
        HRR_tHV1 = 0.1 # Considering only the major Hear release region 
        mask_1 = tf.where(Norm_heat_release > HRR_tHV1)
        redundant_index = mask_1[:, 0]
        #print(redundant_index)

        # Initially the redundant data is being removed from the dataset
        filtered_model_dataset_points = tf.gather(not_normalized, redundant_index, axis=0)
        print("The Appended shape  file:",filtered_model_dataset_points.shape)

        #Appending all the Not_ normalized data points
        if i == 0:
         ppd_FMDP = ppd.create_dataset('FMDP', data=filtered_model_dataset_points, maxshape=(None, 32))
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
print("program end")

