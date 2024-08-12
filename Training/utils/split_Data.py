# -*- coding: utf-8 -*-
import os
join = os.path.join
import shutil
import random
import pandas as pd

path_nii = "/ifs/data/dk3360_gp/projects/DCIS_MRI/" # for E4112, easier for it to be a csv of paths
path_video = None # or specify the path
path_2d = None #'/cbica/home/slavkovk/project_medsam_testing/Data_E4112' # or specify the path

save_path = "/ifs/data/dk3360_gp/kps2152/project_DCIS_MRI/E4112_processed/" # where to save the split up data

#%% split 3D nii data
if path_nii is not None:
    # get a list of the files for all the images and corresponding masks
    img_path = join(path_nii, 'Data', 'E4112')
    mask_path = join(path_nii, 'Masks')
    mask_files = sorted(os.listdir(mask_path)) # these are the mask files themselves
    cases = sorted(os.listdir(img_path)) # these are the case numbers since the folders in img_path are named after the case numbers


    # creat the training, validation, and testing directories
    validation_path = join(save_path, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    
    testing_path = join(save_path, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)

    training_path = join(save_path, 'training')
    os.makedirs(join(training_path, 'images'), exist_ok=True)
    os.makedirs(join(training_path, 'labels'), exist_ok=True)

    # randomly grab 20% of the cases for the val/test split and leave the other 80% for training
    candidates = random.sample(cases, int(len(cases)*0.2))
    training_names = [name for name in cases if name not in candidates]
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]

    
    # move the training, validation, and test files to corresponding folders in save_path
    for name in training_names:
        # pull the image and mask paths from the dataframe
        img_file_i = join(img_path, name, "case_{0}_first_BC.nii.gz".format(name))
        mask_file_i = join(mask_path, "case_{0}_mask.nii.gz".format(name))

        # get the image and mask file names 
        img_name = img_file_i.split("/")[-1]
        mask_name = mask_file_i.split("/")[-1]

        # copy the files to their new locations in validation_path
        shutil.copy(img_file_i, join(training_path, 'images', img_name))
        shutil.copy(mask_file_i, join(training_path, 'labels', mask_name))

    for name in validation_names:
        # pull the image and mask paths from the dataframe
        img_file_i = join(img_path, name, "case_{0}_first_BC.nii.gz".format(name))
        mask_file_i = join(mask_path, "case_{0}_mask.nii.gz".format(name))

        # get the image and mask file names 
        img_name = img_file_i.split("/")[-1]
        mask_name = mask_file_i.split("/")[-1]

        # copy the files to their new locations in validation_path
        shutil.copy(img_file_i, join(validation_path, 'images', img_name))
        shutil.copy(mask_file_i, join(validation_path, 'labels', mask_name))
    
    for name in test_names:
        # pull the image and mask paths from the dataframe
        img_file_i = join(img_path, name, "case_{0}_first_BC.nii.gz".format(name))
        mask_file_i = join(mask_path, "case_{0}_mask.nii.gz".format(name))

        # get the image and mask file names 
        img_name = img_file_i.split("/")[-1]
        mask_name = mask_file_i.split("/")[-1]

        # copy the files to their new locations in validation_path
        shutil.copy(img_file_i, join(testing_path, 'images', img_name))
        shutil.copy(mask_file_i, join(testing_path, 'labels', mask_name))


##% split 2D images
if path_2d is not None:
    img_path = join(path_2d, 'images')
    gt_path = join(path_2d, 'labels')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '.npy'
    gt_suffix = '.npy'
    # split 20% data for validation and testing
    validation_path = join(path_2d, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    
    testing_path = join(path_2d, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_names, int(len(gt_names)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    
    # move validation and testing data to corresponding folders
    for name in validation_names:
        name_vec = name.split('_')
        if img_suffix in name:
            img_name = name_vec[0]+"_img_"+name_vec[2]+"_"+name_vec[3]
            os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
            os.rename(join(gt_path, name), join(validation_path, 'labels', name))

    for name in test_names:
        name_vec = name.split('_')
        if img_suffix in name:
            img_name = name_vec[0]+"_img_"+name_vec[2]+"_"+name_vec[3]
            os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
            os.rename(join(gt_path, name), join(testing_path, 'labels', name))

#%% split video data
if path_video is not None:
    img_path = join(path_video, 'images')
    gt_path = join(path_video, 'labels')
    gt_folders = sorted(os.listdir(gt_path))
    # split 20% videos for validation and testing
    validation_path = join(path_video, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_video, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_folders, int(len(gt_folders)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        os.rename(join(img_path, name), join(validation_path, 'images', name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))
    for name in test_names:
        os.rename(join(img_path, name), join(testing_path, 'images', name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))
