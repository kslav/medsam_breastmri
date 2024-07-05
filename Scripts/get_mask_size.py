# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
#import cc3d
import pandas as pd

### Load path to csv file of imaging paths
img_dirs_csv = "/cbica/home/slavkovk/project_medsam_testing/Data_E4112/e4112_data_training_n238.csv"
img_dirs_df = pd.read_csv(img_dirs_csv)

### Settings
image_size = 1024 # target size to which to resize the images and masks
save_path = "/cbica/home/slavkovk/project_medsam_testing/Data_E4112"

# %% save preprocessed images and masks as npz files
mask_vols = {}
for  idx in range(0,len(img_dirs_df)):  # use the remaining 10 cases for validation
    ### Load a row of the CSV file ###
    case_i = img_dirs_df.loc[idx,'Case']
    img_path_gt = img_dirs_df.loc[idx,'Image']   #ground truth dicom path (probably will use nifti)
    mask_path_gt = img_dirs_df.loc[idx,'Mask']   #ground truth dicom mask path (probably will use nifti)
        
    # ### Load the image and mask corresponding to the paths from the csv file
    # ### Note: SITK loads images as [z, x, y], while nib loads them as [x,y,z]
    # img_nii = nib.load(img_path_gt)    #ground truth image loaded with SITK -> numpy array
    # mask_nii = nib.load(mask_path_gt)  #ground truth mask loaded with SITK -> numpy array
    # #convert to numpy array
    # img_i = np.array(img_nii.dataobj)
    # mask_i = np.array(mask_nii.dataobj)

    img_i = sitk.GetArrayFromImage(sitk.ReadImage(img_path_gt))    #ground truth image loaded with SITK -> numpy array
    mask_i = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_gt))  #ground truth mask loaded with SITK -> numpy array

    # find non-zero slices
    z_index,_,_ = np.where(mask_i > 0)
    z_index = np.unique(z_index)
    mask_vols[case_i]=0
    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        mask_i_crop = mask_i[z_index,:,:]
    
       
        ### Prepare the final training data by resizing the image and mask and saving each slice as .npy file
        vol_sum = 0
        for sli,_ in enumerate(z_index):

            mask_2D = np.squeeze(mask_i_crop[sli,:,:])

            # resize the mask as well
            resize_gt_skimg = transform.resize(
                mask_2D,
                (image_size, image_size),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False)
            
            vol_sum += np.sum(resize_gt_skimg)
    
    mask_vols[case_i]=vol_sum

mask_vols_df = pd.DataFrame.from_dict(mask_vols,orient='index')
mask_vols_df.to_csv(save_path+"/mask_gt_vols_train.csv")





