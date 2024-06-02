#!/usr/bin/env python
"""A dataset object for MRI data that enables compatability with PyTorch."""

from torch.utils.data import Dataset
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import pathlib
import SimpleITK as sitk
from skimage import io, transform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class SegMRIDataset(Dataset):
# PURPOSE: Define a Dataset for MRIs that loads in pairs of ground truth 
# and input pairs from a CSV file. 

# FILE TYPES: Images are all loaded as dicoms from the directories in the CSV 
# file and are loaded as such before being converted to torch Tensors

# TRANSFORMS: NULL for now, consider rotaitons and translations

    # A custom Dataset class must have the following three functions
    def __init__(self, img_dirs_csv, transform=None,which_file='nii'):
        self.img_dirs_csv = img_dirs_csv # CSV of ground truth image and mask pairs of 2D dicoms (ah or niftis)
        self.transform = transform
        self.which_file=which_file
        self.img_dirs_df = pd.read_csv(self.img_dirs_csv)

    def __len__(self):
        return len(self.img_dirs_df) # how many ground truth and input pairs

    def __getitem__(self, idx):

        ### Load the mask and image ###
        img_dirs = self.img_dirs_df # data frame of img_dirs_csv loaded with pandas
        img_path_gt = img_dirs.loc[idx,'Image']   #ground truth dicom path (probably will use nifti)
        mask_path_gt = img_dirs.loc[idx,'Mask']   #ground truth dicom mask path (probably will use nifti)
        
        if self.which_file=='dcm': # if dicom format...
            img_gt = sitk.GetArrayFromImage(sitk.ReadImage(img_path_gt))    #ground truth image loaded with SITK -> numpy array
            mask_gt = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_gt))  #ground truth mask loaded with SITK -> numpy array
            #print("Image and mask shapes are ", img_gt.shape, " and ", mask_gt.shape)
        elif self.which_file=='nii': # if nifti format...
            img_nii = nib.load(img_path_gt)    #ground truth image loaded with SITK -> numpy array
            mask_nii = nib.load(mask_path_gt)  #ground truth mask loaded with SITK -> numpy array

            #convert to numpy array
            img_gt = np.array(img_nii.dataobj)
            mask_gt = np.array(mask_nii.dataobj)
            

        ### Process image and mask to be 3-channel and normalized ###

        # normalize the image
        lower_bound, upper_bound = np.percentile(img_gt[img_gt > 0], 0.5), np.percentile(img_gt[img_gt > 0], 99.5)
        image_data_pre = np.clip(img_gt, lower_bound, upper_bound)
        image_data_pre = ((image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre))* 255.0)
        image_data_pre[img_gt == 0] = 0
        img_gt = image_data_pre # re-name back to img_gt

        dims = img_gt.shape
        if len(dims) == 2:
            #img_gt = img_gt.transpose() # fix the view
            img_3c = np.repeat(img_gt[:, :, None], 3, axis=-1) #copy slice along 3rd dim
            
            #3mask_gt = mask_gt.transpose() #fix the view
            mask_3c = mask_gt[:, :, None] #repeat for mask
        
        elif len(dims) == 3:
            ### For now, we're selecting the central slice since training data is 3D vol... ###
            #img_gt = img_gt.transpose(1,0,2) # fix the view
            img_gt = np.squeeze(img_gt[:,:,dims[2]//2])
            img_3c = np.repeat(img_gt[:, :, None], 3, axis=-1)

            #mask_gt = mask_gt.transpose(1,0,2) #fix the view
            mask_gt = np.squeeze(mask_gt[:,:,dims[2]//2])
            mask_3c = mask_gt[:,:,None] #same for mask
        else:
            print("Something's wrong with the image size!")

        
        H, W = dims[1],dims[0] # Height and width, respectively, of image (also applies to mask)
        
        # resize the image and mask to 1024x1024 (required for inputs into SAM)
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)#bicubic interp
        mask_1024 = transform.resize(mask_3c, (1024, 1024), order=0, preserve_range=True, anti_aliasing=True).astype(np.uint8) #nearest neighbor interp
        #print("Image and mask shapes after preprocessing (before resize): ", img_1024.shape, " and ", mask_1024.shape)
        # [obsolete, already done above] normalize the image to a range of [0, 1]
        #img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None) 
        
        # permute the image and mask to have shape (3, H, W)
        img_1024 = img_1024.transpose(2, 0, 1)
        mask_1024 = mask_1024.transpose(2, 0, 1)
        bboxes = np.array([200, 200, 900, 900])
        #print("Number of unique values in mask...", np.unique(mask_1024))
        assert len(np.unique(mask_1024))<=2, "ground truth mask should have just 2 values: 0, 1"

        # Build the bounding box!

        if self.transform is not None:
            #(Note that the ToTensor transformation -- as self.transform -- will yield image values in range 0 to 1 ONLY for images with range [0, 255])
            #img_1024 = self.transform(img_1024.astype("float32")) #apply any transformations during training to both GT and inp
            #mask_1024 = self.transform(mask_1024.astype("float32"))
            img_1024 = torch.tensor(img_1024).float()
            mask_1024 = torch.tensor(mask_1024).long()
            bboxes = torch.tensor(bboxes).float()


        #print("Image and mask shapes are ", img_1024.shape, " and ", mask_1024.shape)
        return img_1024, mask_1024, bboxes # each "item" is a pair of ground truth image and mask with shape [3,H,W]; dataloader automatically adds batch dim out front
