
from skimage import io, transform
from torch.utils.data import Dataset
import os
import glob
import numpy as np
join = os.path.join

class NpyDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform=transform
        self.mask_path = join(data_root, "labels")
        self.img_path = join(data_root, "images")
        self.mask_path_files = sorted(glob.glob(join(self.mask_path, "**/*.npy"), recursive=True))
        self.img_path_files = sorted(glob.glob(join(self.img_path, "**/*.npy"), recursive=True))

        print(f"number of images: {len(self.mask_path_files)}")

    def __len__(self):
        return len(self.mask_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.img_path_files[index])
        mask_name = os.path.basename(self.mask_path_files[index])

        #check that img and mask correspond (same case and slice number)
        img_name_vec = img_name.split('_')
        mask_name_vec = mask_name.split('_')

        print("img_name: ", img_name, " mask_name: ", mask_name)

        #img and mask file names formatted as [case#]_[img or mask]_sli_[sli#].npy
        assert (img_name_vec[0]==mask_name_vec[0]) and (img_name_vec[2:]==mask_name_vec[2:]), "Image and mask files don't correspond!"
        
        #load the image
        img_1024 = np.load(self.img_path_files[index], "r", allow_pickle=True)  # (1024, 1024, 3)

        assert (np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0), "image should be normalized to [0, 1]"
        
        #load the mask
        mask_1024 = np.load(self.mask_path_files[index], "r", allow_pickle=True)  # multiple labels [0, 1,4,5...], (256,256)
        # convert the shape to (3, H, W)

        #fixed bounding box
        bboxes = np.array([200, 200, 900, 900])

        if self.transform is not None:
            #(Note that the ToTensor transformation -- as self.transform -- will yield image values in range 0 to 1 ONLY for images with range [0, 255])
            img_1024 = self.transform(img_1024.astype("float32")) #apply any transformations during training to both GT and inp
            mask_1024 = self.transform(mask_1024.astype("float32"))
        
        #print("Image and mask shapes are ", img_1024.shape, " and ", mask_1024.shape)
        return img_1024, mask_1024, bboxes
