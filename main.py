#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

### setup environment ###
import numpy as np
import os

join = os.path.join
from test_tube import HyperOptArgumentParser
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from segment_anything import sam_model_registry
from Training.model import MedSAM
from Training.datasets import SegMRIDataset
from torchvision import transforms
import argparse
import random
from datetime import datetime
import shutil
import glob


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
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint) #See MedSAM_Inference.py for pretrained model attributes
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder).to(device)
    
    medsam_model.train() # set mode of medsam_model to train

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
    best_loss = 1e10
    
    ### Establish dataset and dataloader ###
    train_dataset = SegMRIDataset(args.train_data_csv, transform=transforms.ToTensor(),which_file='nii')
    print("Number of training samples: ", train_dataset.__len__())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

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
        epoch_loss = 0
        for step, (img_gt, mask_gt, boxes) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # check which device the img, mask, and boxes are on
            print("Devices are...", img_gt.device, mask_gt.device, boxes.device)
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

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        
        ### save the latest model ###
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        ### save the best model ###
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))


if __name__ == "__main__":
    
    ### Set up parser and establish args ###

    usage_str = 'usage: %(prog)s [options]'
    description_str = 'fine tune medsam'
    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='grid_search')
    
    parser.json_config('--config', default=None)
    parser.add_argument("--train_data_csv", action='store',dest='train_data_csv',type=str, default="data/npy/CT_Abd", help="path to CSV with file paths (img and mask GT)")
    parser.add_argument("--task_name", action='store',dest='task_name',type=str, default="MedSAM-ViT-B")
    parser.add_argument("--model_type", action='store',dest='model_type',type=str, default="vit_b")
    parser.add_argument("--checkpoint", action='store',dest='checkpoint',type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument("--work_dir", action='store',dest='work_dir',type=str, default="./work_dir")
    
    # Dataloader parameters
    parser.add_argument("--num_epochs", action='store',dest='num_epochs',type=int, default=1000)
    parser.add_argument("--batch_size", action='store',dest='batch_size',type=int, default=2)
    parser.add_argument("--num_workers", action='store',dest='num_workers', type=int, default=0)
    
    # Optimizer parameters
    parser.add_argument("--weight_decay", action='store',dest='weight_decay',type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", action='store',dest='lr',type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--use_wandb", action='store', dest='use_wandb', type=bool,default=False, help="use wandb to monitor training")
    parser.add_argument("--use_amp", action='store', dest='use_amp', type=bool,default=False, help="use amp")
    parser.add_argument("--resume", action='store',dest='resume',type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device",action='store',dest='device', type=str, default="cuda:0")
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.empty_cache()

    if args.use_wandb:
        import wandb

        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.train_data_csv,
                "model_type": args.model_type,
            },
        )
    # checking pytorch + cuda compatibility:
    #foo = torch.tensor([1,2,3])
    #foo = foo.to(args.device)

    main_train(args)
