
from skimage import io, transform
from torch.utils.data import Dataset
import torch
import os
import glob
import numpy as np
join = os.path.join

class NpyDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root #directory in which data lives
        self.transform=transform
        self.mask_path = join(data_root, "labels") #expect masks in "labels" folder
        self.img_path = join(data_root, "images") #expect images in "images" folder
        self.mask_path_files = sorted(glob.glob(join(self.mask_path, "**/*.npy"), recursive=True)) #get a sorted list of mask files
        self.img_path_files = sorted(glob.glob(join(self.img_path, "**/*.npy"), recursive=True)) #get a sorted list of image files

        print(f"number of images: {len(self.mask_path_files)}")

    def __len__(self):
        return len(self.mask_path_files)

    def __getitem__(self, index):

        # Load npy image (1024, 1024, 3), [0,1], preprocessed using /utils/convert_3Dto2D.py
        img_name = os.path.basename(self.img_path_files[index])
        mask_name = os.path.basename(self.mask_path_files[index])

        #check that img and mask correspond (same case and slice number)
        img_name_vec = img_name.split('_')
        mask_name_vec = mask_name.split('_')

        #CHECK: print("img_name: ", img_name, " mask_name: ", mask_name)

        # Assert: image and mask file names formatted as [case#]_[img or mask]_sli_[sli#].npy should correspond
        assert (img_name_vec[0]==mask_name_vec[0]) and (img_name_vec[2:]==mask_name_vec[2:]), "Image and mask files don't correspond!"
        
        # Load the image
        img_1024 = np.load(self.img_path_files[index], "r", allow_pickle=True)  # (1024, 1024, 3)

        # Assert: image should have values between 0 and 1
        assert (np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0), "image should be normalized to [0, 1]"
        
        # Load the mask
        mask_1024 = np.load(self.mask_path_files[index], "r", allow_pickle=True)  # just two labels [0,1]
        # Convert the shape to (3, H, W)
        img_1024 = img_1024.transpose(2, 0, 1)
        mask_1024 = mask_1024.transpose(2, 0, 1)
        
        #fixed bounding box
        bboxes = np.array([200, 200, 900, 900])

        if self.transform is not None:
            # FIXME: Right now this code block just converts the npy arrays to tensors, no matter what transform you feed it
            #(Note that the ToTensor transformation -- as self.transform -- will yield image values in range 0 to 1 ONLY for images with range [0, 255])
            #img_1024 = self.transform(img_1024.astype("float32")) #apply any transformations during training to both GT and inp
            #mask_1024 = self.transform(mask_1024.astype("float32"))

            img_1024 = torch.tensor(img_1024).float()
            mask_1024 = torch.tensor(mask_1024).long()
            bboxes = torch.tensor(bboxes).float()
        
        #CHECK: print("Image and mask shapes are ", img_1024.shape, " and ", mask_1024.shape)
        return img_1024, mask_1024, bboxes
