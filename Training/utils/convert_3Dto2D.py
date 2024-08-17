# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
join = os.path.join
from skimage import transform
#import cc3d
from scipy.ndimage import rotate
import nibabel as nib

### Load path to csv file of imaging paths
data_path = "/home/kps2152/project_medSAM_testing/Data/E4112/training"
save_path = data_path

### Load the CSV file with orientation and acquisition info for each case so we know how to rotate the images
img_info_csv = "/home/kps2152/project_medSAM_testing/Data/E4112/DCIS_R01_first_BC_resolution_info.csv"
img_info_df = pd.read_csv(img_info_csv)
img_info_df = img_info_df['Case','Orientation']

### Settings
save_cropped_niis = False #whether or not to save the niis of the slice-wise cropped img and mask for inspection
image_size = 1024 # target size to which to resize the images and masks

### Get the list of image and label .nii.gz files in data_path
img_files = sorted(os.listdir(join(data_path,'images')))
mask_files = sorted(os.listdir(join(data_path,'labels')))

# %% save preprocessed images and masks as npz files
for  i in range(0,5):#len(img_files)):  
    # load the image and mask for case i
    img_i = sitk.GetArrayFromImage(sitk.ReadImage(img_files[i]))    #ground truth image loaded with SITK -> numpy array
    mask_i = sitk.GetArrayFromImage(sitk.ReadImage(mask_files[i]))  #ground truth mask loaded with SITK -> numpy array
    
    # get the orientation info from img_info_df
    orien_i = img_info_df.iloc[i, 1] 
    case_i = img_info_df.iloc[i, 0] 

    # Check that case_i is the same as the case number in img_files[i]
    filename = img_files[i]
    filename_vec = filename.split('/')
    filename_nii = filename_vec[-1]
    filename_nii_vec = filename_nii.split('_')
    assert int(filename_nii_vec[1])==case_i, "Case number from CSV and case number from image file aren't the same..."
    
    # Check that the case numbers in the img and mask file names are the same...
    filename2 = mask_files[i]
    filename2_vec = filename2.split('/')
    filename2_nii = filename2_vec[-1]
    filename2_nii_vec = filename2_nii.split('_')
    assert int(filename2_nii_vec[1])==int(filename_nii_vec[1]), "Case number in image and mask file names should be the same..."
    

    # rotate the image about the slice axis so that the chest wall is along the bottom 
    if orien_i == 'T':
        img_i = rotate(img_i, angle=180, axes=(2, 1), reshape=False)
        mask_i = rotate(img_i, angle=180, axes=(2, 1), reshape=False)
    if orien_i == 'R':
        img_i = rotate(img_i, angle=270, axes=(2, 1), reshape=False)
        mask_i = rotate(img_i, angle=270, axes=(2, 1), reshape=False)

    # find non-zero slices
    z_index,_,_ = np.where(mask_i > 0)
    z_index = np.unique(z_index)

    #sanity check stuff:
    print("image size = ", img_i.shape)
    print("mask size = ", mask_i.shape)
    print("z_index = ", z_index)
    print("image max and min are ", np.max(img_i), " ", np.min(img_i))


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
            sitk.WriteImage(img_roi_sitk, join(save_path, "sanitycheck_case_img_"+str(idx)+".nii"))
            sitk.WriteImage(gt_roi_sitk, join(save_path, "sanitycheck_case_mask_"+str(idx)+".nii"))
       
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



