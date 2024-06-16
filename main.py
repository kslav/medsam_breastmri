#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

### setup environment ###
import numpy as np
import os
import wandb

join = os.path.join
from test_tube import HyperOptArgumentParser
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import monai
from segment_anything import sam_model_registry
from Training.model import MedSAM
from Training.datasets import SegMRIDataset
from Training.datasets import NpyDataset
from torchvision import transforms
import argparse
import random
from datetime import datetime
import shutil
import glob
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def main_train(args):
    ### Set up save directories ###
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id) # where the final model will be saved!
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))) 

    ### Set up the model for training, namely load pretrained model ###
    device = torch.device(args.device)
    print("Device is ...", device)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint) #See MedSAM_Inference.py for pretrained model attributes
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder).to(device)

    ### Set optimizer ###
    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay)

    ### Print out attributes of medsam_model ###
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    
    ### Establish loss functions ###
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    val_losses = []
    best_loss = 1e10
    
    ### Establish dataset and dataloader ###
    if args.which_dataloader=="npy":
        train_dataset = NpyDataset(args.train_data_paths, transform=transforms.ToTensor()) #Note, bypassing ToTensor() because it rescales data, just want to cast to tensor
        val_dataset = NpyDataset(args.val_data_paths, transform=transforms.ToTensor()) #Note, bypassing ToTensor() because it rescales data, just want to cast to tensor
    elif args.which_dataloader=="mri":
        train_dataset = SegMRIDataset(args.train_data_paths, transform=transforms.ToTensor(),which_file='nii') #Note, bypassing ToTensor() because it rescales data, just want to cast to tensor
        val_dataset = SegMRIDataset(args.train_data_paths, transform=transforms.ToTensor(),which_file='nii')    

    print("Number of training samples: ", train_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False,)

    ### Training loop ###
    start_epoch = 0
    if args.resume is not None: # if resuming training from last epoch the training left off at....
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device) #load the last checkpoint
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"]) #load the model from the checkpoint...
            optimizer.load_state_dict(checkpoint["optimizer"]) #load the optimizer from the checkpoint
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        print("Current epoch is...", epoch)
        epoch_loss = 0
        val_epoch_loss=0

        ### TRAINING STEP ### 
        medsam_model.train()
        for step, (img_gt, mask_gt, boxes) in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()
            # check which device the img, mask, and boxes are on
            #print("Devices are...", img_gt.device, mask_gt.device, boxes.device)
            #print("Okay printing shapes here...", img_gt.shape, mask_gt.shape)
            # create bounding box here that is the size of img_gt
            boxes_np = boxes.detach().cpu().numpy()
            
            img_gt, mask_gt = img_gt.to(device), mask_gt.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(img_gt, boxes_np)
                    loss = seg_loss(medsam_pred, mask_gt) + ce_loss(
                        medsam_pred, mask_gt.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(img_gt, boxes_np)
                loss = seg_loss(medsam_pred, mask_gt) + ce_loss(medsam_pred, mask_gt.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

            # log images in wandb every n steps 
            if args.use_wandb and step%args.log_frequency==0: 
                with torch.no_grad():
                    # pick a random image in the batch
                    item_idx = np.random.randint(0,args.batch_size) 
                    
                    # get the image and gt mask
                    img_idx = img_gt[item_idx,...]
                    mask_gt_idx = mask_gt[item_idx,...]
                    medsam_pred_idx = medsam_pred[item_idx,...] 

                    # move img and mask_gt to cpu and convert to numpy
                    box = np.squeeze(boxes_np[item_idx,...])
                    img_np = img_idx.permute(1, 2, 0).detach().cpu().numpy()
                    mask_np = np.squeeze(mask_gt_idx.detach().cpu().numpy())

                    # convert the medsam_pred to [0,1] before moving to cpu and converting to numpy
                    pred_sig = torch.sigmoid(medsam_pred_idx[None,...])
                    pred_sig_np = pred_sig.squeeze().cpu().numpy()
                    pred_np = (pred_sig_np > 0.5).astype(np.uint8)


                    # make figure for logging
                    mask_arr_forFig = [mask_np,pred_np]
                    fig = make_image_for_logging(img_np,mask_arr_forFig,box)

                    # compute dice score as our quality metric
                    #FOR DEBUG: print("unique vals in pred_np are ", np.unique(pred_np))
                    #FOR DEBUG: print("unique vals in mask_gt are ", np.unique(mask_np))
                    train_dice = np.sum(pred_np[mask_np==1])*2.0 / (np.sum(pred_np) + np.sum(mask_np))
                    # Log the figure we made
                    wandb.log({"Train_Comparison": wandb.Image(fig)})
                    # Log the step-level metrics here:
                    wandb.log({"train_loss_step": loss.item()})
                    wandb.log({"train_dice_step": train_dice})
                
        
        # log epoch-level training metrics:
        epoch_loss /= step
        losses.append(epoch_loss)
        # Log epoch-level metrics here
        if args.use_wandb:
            # Log the image with the mask overlay
            wandb.log({"epoch": epoch, "train_loss_epoch": epoch_loss})
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        ### VALIDATION STEP ###
        medsam_model.eval()
        with torch.no_grad():
            for step, (img_gt_val, mask_gt_val, boxes_val) in enumerate(tqdm(val_dataloader)):
                boxes_np = boxes_val.detach().cpu().numpy()
                img_gt_val, mask_gt_val = img_gt_val.to(device), mask_gt_val.to(device)
                medsam_pred = medsam_model(img_gt_val, boxes_np) #prediction

                val_loss = seg_loss(medsam_pred, mask_gt_val) + ce_loss(medsam_pred, mask_gt_val.float()) #loss
                val_epoch_loss += val_loss.item()

                if args.use_wandb and step%args.log_frequency==0: 
                    # pick a random image in the batch
                    item_idx = np.random.randint(0,args.batch_size) 
                    
                    # get the image and gt mask
                    img_idx = img_gt_val[item_idx,...]
                    mask_gt_idx = mask_gt_val[item_idx,...]
                    medsam_pred_idx = medsam_pred[item_idx,...] 

                    # move img and mask_gt to cpu and convert to numpy
                    box = np.squeeze(boxes_np[item_idx,...])
                    img_np = img_idx.permute(1, 2, 0).detach().cpu().numpy()
                    mask_np = np.squeeze(mask_gt_idx.detach().cpu().numpy())

                    # convert the medsam_pred to [0,1] before moving to cpu and converting to numpy
                    pred_sig = torch.sigmoid(medsam_pred_idx[None,...])
                    pred_sig_np = pred_sig.squeeze().cpu().numpy()
                    pred_np = (pred_sig_np > 0.5).astype(np.uint8)

                    # make figure for logging
                    mask_arr_forFig = [mask_np,pred_np]
                    fig = make_image_for_logging(img_np,mask_arr_forFig,box)

                    # compute dice score as our quality metric
                    val_dice = np.sum(pred_np[mask_np==1])*2.0 / (np.sum(pred_np) + torch.sum(mask_np))

                    # Log the figure we made
                    wandb.log({"Val_Comparison": wandb.Image(fig)})
                    # Log the step-level metrics here:
                    wandb.log({"val_loss_step": val_loss.item()})
                    wandb.log({"val_dice_step": val_dice})


            # log epoch-level validation metrics:
            val_epoch_loss /= step
            val_losses.append(val_epoch_loss)
            # Log epoch-level metrics here
            if args.use_wandb:
                # Log the image with the mask overlay
                wandb.log({"val_loss_epoch": val_epoch_loss})
            
        ### save the latest model ###
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss_epoch": losses,
            "val_loss_epoch": val_losses,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        ### save the best model ###
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss_epoch": losses,
                "val_loss_epoch": val_losses,
            }

            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

### The following four functions are for visualizating figures in WandB

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

def make_image_for_logging(img, mask_arr,box):
    # PURPOSE: To design a figure where we have the image with ground truth mask overlayed on the left
    # and the prediction mask overlayed on the right, side by side. Once we've made this image, we store
    # it in the buffer, retrieve it from the buffer, and then log it with WandB.
    # img: imag of size 1024x1024x3, randomly selected from current batch
    # mask_arr: array of size 2, containing ground truth [0] and prediction masks [1]

    # make the figure with matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    show_mask(mask_arr[0], ax[0])
    ax[0].set_title("Input Image and Ground Truth")
    ax[1].imshow(img)
    show_mask(mask_arr[1], ax[1])
    show_box(box, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    plt.tight_layout()

    buf = BytesIO()  # create a BytesIO buffer
    fig.savefig(buf, format='png')  # save the plot to the buffer in PNG format
    buf.seek(0)  # rewind the buffer to the beginning
    img_out = Image.open(buf)  # open the image from the buffer

    plt.close(fig)  # close the figure to free memory
    return img_out


if __name__ == "__main__":
    
    ### Set up parser and establish args ###

    usage_str = 'usage: %(prog)s [options]'
    description_str = 'fine tune medsam'
    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='grid_search')
    
    # Initialization parameters
    parser.json_config('--config', default=None)
    parser.add_argument("--task_name", action='store',dest='task_name',type=str, default="MedSAM-ViT-B")
    parser.add_argument("--model_type", action='store',dest='model_type',type=str, default="vit_b")
    parser.add_argument("--checkpoint", action='store',dest='checkpoint',type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument("--work_dir", action='store',dest='work_dir',type=str, default="./work_dir")
    
    # Dataloader parameters
    parser.add_argument("--which_dataloader", action='store',dest='which_dataloader',type=str, default="npy",help="[npy,mri] choose whether to load npys from folder or niftis from csv paths")
    parser.add_argument("--train_data_paths", action='store',dest='train_data_paths',type=str, default="", help="path to dir or CSV with file paths (img and mask GT)")
    parser.add_argument("--val_data_paths", action='store',dest='val_data_paths',type=str, default="", help="path to dir or CSV with file paths (img and mask GT)")
    parser.add_argument("--num_epochs", action='store',dest='num_epochs',type=int, default=1000)
    parser.add_argument("--batch_size", action='store',dest='batch_size',type=int, default=2)
    parser.add_argument("--num_workers", action='store',dest='num_workers', type=int, default=0)
    
    # Optimizer parameters
    parser.add_argument("--weight_decay", action='store',dest='weight_decay',type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", action='store',dest='lr',type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--use_amp", action='store', dest='use_amp', type=bool,default=False, help="use amp")
    parser.add_argument("--resume", action='store',dest='resume',type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device",action='store',dest='device', type=str, default="cuda:0")

    # Logging parameters
    parser.add_argument("--use_wandb", action='store', dest='use_wandb', type=bool,default=False, help="use wandb to monitor training")
    parser.add_argument("--log_frequency", action='store',dest='log_frequency',type=int, default=100, help="define N for logging every N steps to save on memory")

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.empty_cache()

    if args.use_wandb:
        import wandb

        #wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "train_data_path": args.train_data_paths,
                "val_data_path": args.val_data_paths,
                "model_type": args.model_type,
            },
        )
    # checking pytorch + cuda compatibility:
    #foo = torch.tensor([1,2,3])
    #foo = foo.to(args.device)

    main_train(args)
