#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from test_tube import HyperOptArgumentParser
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
from Training.model import MedSAM
from Training.datasets import SegMRIDataset
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob



def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processces")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    
    ### Set up DDP related stuff ###
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    # device = torch.device("cuda:{}".format(gpu))
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    ### Set up model by loading from checkpoint ###
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()
    
    ### Print out cuda memory related things before DDP initialization ###
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system("nvidia-smi")

    ### Set up model for distributed training
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    ### Print out cuda memory related things after DDP initialization###
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("After DDP initialization:")
        os.system("nvidia-smi")

    medsam_model.train() # set mode of medsam_model to train

    ### Print out attributes of medsam_model ###
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    ### Setting up optimiser ###
    # only optimize the parameters of image encodder, mask decoder, do not update prompt encoder
    # img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    img_mask_encdec_params = list(medsam_model.module.image_encoder.parameters()) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay)
    
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    
    ### Establish loss functions ###
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    
    ### Establish dataset and dataloader ###
    train_dataset = SegMRIDataset(args.train_data_csv, transform=transforms.ToTensor(),which_file='nii')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    ### Training loop ###
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        for step, (image, gt2D, boxes, _) in enumerate(
            tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]")
        ):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            # image, gt2D = image.to(device), gt2D.to(device)
            image, gt2D = image.cuda(), gt2D.cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
                # Gradient accumulation
                if args.grad_acc_steps > 1:
                    loss = (
                        loss / args.grad_acc_steps
                    )  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with medsam_model.no_sync():
                            loss.backward()  ## calculate the gradient only
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if step > 10 and step % 100 == 0:
                if is_main_host:
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint,
                        join(model_save_path, "medsam_model_latest_step.pth"),
                    )

            epoch_loss += loss.item()
            iter_num += 1

            # if rank % ngpus_per_node == 0:
            #     print('\n')
            #     os.system('nvidia-smi')
            #     print('\n')

        # Check CUDA memory usage
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
            1
        ] / (1024**3)
        print("\n")
        print(f"[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb")
        print(f"[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb")
        print(
            f"[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb"
        )
        print("\n")

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        # save the model checkpoint
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

            ## save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
        torch.distributed.barrier()

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


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

    #parser.add_argument(
    #    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
    #)
    #parser.add_argument("-pretrain_model_path", type=str, default="")
    
    # train
    parser.add_argument("--num_epochs", action='store',dest='num_epochs',type=int, default=1000)
    parser.add_argument("--batch_size", action='store',dest='batch_size',type=int, default=2)
    parser.add_argument("--num_workers", action='store',dest='num_workers', type=int, default=0)
    
    # Optimizer parameters
    parser.add_argument("--weight_decay", action='store',dest='weight_decay',type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", action='store',dest='lr',type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--use_wandb", action='store', dest='use_wandb', type=bool,default=False, help="use wandb to monitor training")
    parser.add_argument("--use_amp", action='store', dest='use_amp', type=bool,default=False, help="use amp")
    
    # Distributed training args
    parser.add_argument("--world_size", action='store',dest='world_size', type=int, help="world size")
    parser.add_argument("--node_rank", action='store',dest='node_rank', type=int, default=0, help="Node rank")
    parser.add_argument("--bucket_cap_mb", action='store',dest='bucket_cap_mb', type=int, default=25, help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)")
    parser.add_argument("--grad_acc_steps", action='store',dest='grad_acc_steps', type=int, default=1, help="Gradient accumulation steps before syncing gradients for backprop")
    parser.add_argument("--resume", action='store',dest='resume',type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--init_method", action='store',dest='init_method',type=str, default="env://")

    args = parser.parse_args()

    ### Log to weights and biases if flag set to true ###
    if args.use_wandb:
        import wandb

        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
            },
        )

    # set seeds
    torch.manual_seed(2023)
    torch.cuda.empty_cache()

    main(args)
