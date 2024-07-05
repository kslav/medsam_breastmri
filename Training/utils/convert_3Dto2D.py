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
img_dirs_csv = "/cbica/home/slavkovk/project_medsam_testing/Data_E4112/e4112_data_validation_n60.csv"
img_dirs_df = pd.read_csv(img_dirs_csv)

### Settings
save_cropped_niis = False #whether or not to save the niis of the cropped img and mask for inspection
image_size = 1024 # target size to which to resize the images and masks
save_path = "/cbica/home/slavkovk/project_medsam_testing/Data_E4112/split_patientwise/validation"

# %% save preprocessed images and masks as npz files
for  idx in range(0,5):#len(img_dirs_df)):  
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

    #sanity check stuff:
    #print("image size = ", img_i.shape)
    #print("mask size = ", mask_i.shape)
    #print("z_index = ", z_index)
    #print("image max and min are ", np.max(img_i), " ", np.min(img_i))


    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        mask_i_crop = mask_i[z_index,:,:]
        
        # preprocess the image before cropping
        lower_bound, upper_bound = np.percentile(img_i[img_i > 0], 0.5), np.percentile(img_i[img_i > 0], 99.5)
        image_data_pre = np.clip(img_i, lower_bound, upper_bound)
        image_data_pre = ((image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre))* 255.0)
        image_data_pre[img_i == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_i_crop = image_data_pre[z_index,:,:]
        
        ### Save the cropped image and mask as nii's for quality check in ITK-Snap
        if save_cropped_niis:
            img_roi_sitk = sitk.GetImageFromArray(img_i_crop)
            gt_roi_sitk = sitk.GetImageFromArray(mask_i_crop)
            sitk.WriteImage(img_roi_sitk, "/cbica/home/slavkovk/img_"+str(idx)+".nii")
            sitk.WriteImage(gt_roi_sitk,"/cbica/home/slavkovk/mask_"+str(idx)+".nii")
       
        ### Prepare the final training data by resizing the image and mask and saving each slice as .npy file
        for sli,_ in enumerate(z_index):
            
            # check if the desired paths even exist:
            img_path_final = join(save_path,"images",str(case_i)+"_img_sli_"+str(sli).zfill(3)+".npy")
            mask_path_final = join(save_path,"labels",str(case_i)+"_mask_sli_"+str(sli).zfill(3)+".npy")
            if (os.path.isfile(img_path_final) and os.path.isfile(mask_path_final))==False:
                img_2D = np.squeeze(img_i_crop[sli, :, :])
                mask_2D = np.squeeze(mask_i_crop[sli,:,:])
                img_3c = np.repeat(img_2D[:, :, None], 3, axis=-1)
                mask_3c = mask_2D[...,None]

                resize_img_skimg = transform.resize(
                    img_3c,
                    (image_size, image_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,)
                
                # normalize the resized image
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None)  # normalize to [0, 1], (H, W, 3)
                
                # resize the mask as well
                resize_gt_skimg = transform.resize(
                    mask_3c,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False)
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape[:2]

                
                np.save(img_path_final,resize_img_skimg_01)
                np.save(mask_path_final,resize_gt_skimg)



