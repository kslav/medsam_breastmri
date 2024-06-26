# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Training.datasets import NpyDataset
import pandas as pd
import argparse
from tqdm import tqdm
# update 

# adapted from Ma et al. /MedSAM/MedSAM_Inference.py
# PURPOSE: generate baseline segmentations
# Approach: Loop through imgaes and mask and create bounding boxes based on ground truth segmentations

def make_gt_box(mask, pad_size):
    #PURPOSE: create a ground truth bounding box from a ground truth mask
    # pad_size is the percent (w.r.t the width of the ground truth mask) of padding to add (e.g. 0.2, 0.3, etc.)

    # find min and max x and y of the lesion in the mask
    x_idx, y_idx = np.where(mask > 0)
    x_min, y_min, x_max, y_max = np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)

    # get the width and height of the box that tightly surrounds the ground truth lesion
    w, h = x_max-x_min, y_max - y_min

    x_min_pad = np.uint8(x_min - w*pad_size)
    x_max_pad = np.uint8(x_max + w*pad_size)
    y_min_pad = np.uint8(y_min - h*pad_size)
    y_max_pad = np.uint8(y_max + h*pad_size)

    # add the padding
    box = np.array([x_min_pad, y_min_pad, x_max_pad, y_max_pad])
    
    return box



def show_mask(mask, ax, random_color=False):
    # Borrowed from MedSAM_Inference.py by the original authors of MedSAM, Ma et al.
    # mask is 2D
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    # Borrowed from MedSAM_Inference.py by the original authors of MedSAM, Ma et al.
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))

def make_image_for_logging(img, mask_arr,box,dice_score, save_file):
    # PURPOSE: To design a figure where we have the image with ground truth mask overlayed on the left
    # and the prediction mask overlayed on the right, side by side. Once we've made this image, we store
    # it in the buffer, retrieve it from the buffer, and then log it with WandB.
    # img: imag of size 1024x1024x3, randomly selected from current batch
    # mask_arr: array of size 2, containing ground truth [0] and prediction masks [1]
    # dice_score: scalar value of dice score corresponding to img and predicted mask

    # make the figure with matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    show_mask(mask_arr[0], ax[0])
    ax[0].set_title("Input Image and Ground Truth")
    ax[1].imshow(img)
    show_mask(mask_arr[1], ax[1])
    show_box(box, ax[1])
    ax[1].set_title("MedSAM Segmentation - DICE={dice}".format(dice=round(dice_score,3)))
    plt.tight_layout()

    fig.savefig(save_file)  # save the plot to the buffer in PNG format
    plt.close(fig)  # close the figure to free memory
    


@torch.no_grad()
def medsam_inference(medsam_model, img, box, H, W):
    #PURPOSE: run inference and output a mask that is of the original image size, H and W
    # (recall that images are resized to 1024x1024 for input into medsam)
    # returns numpy array mask prediction

    medsam_pred = medsam_model(img, box) #prediction
    pred_sig_np = medsam_pred.squeeze().cpu().numpy()
    pred_np = (pred_sig_np > 0.5).astype(np.uint8)

    return pred_np


#### Set up the arguments ####
parser = argparse.ArgumentParser(description="run inference on testing set based on MedSAM")
parser.add_argument("--data_path", action='store',dest='data_path',type=str, default="/cbica/home/slavkovk/project_medsam_testing/Data_E4112/training", help="path to dir with images and masks")
parser.add_argument("--seg_path", action='store', dest='seg_path',type=str, default="/cbica/home/slavkovk/project_medsam_testing/Data_E4112/baselines", help="path to dir in which to store predictions")
parser.add_argument("--device",action='store',dest='device', type=str, default="cuda:0")
parser.add_argument("--checkpoint", action='store',dest='checkpoint',type=str, default="/cbica/home/slavkovk/project_medsam_testing/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
parser.add_argument("--pad_size",action='store',dest='pad_size',type=float,default=0.2)
args = parser.parse_args()

#### Load the model and set the device ####
device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

### Create Dataloader ###
dataSet = NpyDataset(args.data_path, transform=transforms.ToTensor())
dataLoader = DataLoader(dataSet,batch_size=1,shuffle=False,num_workers=2,pin_memory=False)

### Loop through images and masks, creating a gt bounding box and running inference ####

dice_scores = {}
save_every_n_steps = 100
for step, (img_gt, mask_gt, _, img_name) in enumerate(tqdm(dataLoader)):

    #put the image on the same device as the model
    img_gt, mask_gt = img_gt.to(device), mask_gt.to(device)

    # create the ground truth boudning box
    box = make_gt_box(mask_gt, args.pad_size)
    
    # get the medsam prediction
    mask_pred = medsam_inference(img_gt, box)

    # compute the dice score
    mask_gt = mask_gt.squeeze().cpu().numpy()
    dice_step = np.sum(mask_pred[mask_gt==1])*2.0 / (np.sum(mask_pred) + np.sum(mask_gt))
    dice_scores[img_name]=dice_step

    # save a figure every save_every_n_steps for visualization:
    if step % save_every_n_steps == 0:
        img_temp = img_gt[0,...]
        img_np = img_temp.permute(1,2,0).detach().cpu().numpy()

        img_name_vec = img_name.split('_')
        save_file = join(args.seg_path,img_name_vec[0]+"_"+"baseline_"+str(args.pad_size)+".png")
        make_image_for_logging(img_np, [mask_gt, mask_pred],box,dice_step, save_file)

# save the dictionary of dice scores
df = pd.DataFrame.from_dict(dice_scores,orient="index")
df.to_csv(join(args.seg_path,"dice_scores_baseline.csv"),index_col=0)



