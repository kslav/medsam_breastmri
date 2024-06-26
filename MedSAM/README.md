# MedSAM
This README.md is modified from the official MedSAM repo README, created by Ma et al., and located at https://github.com/bowang-lab/MedSAM. Acknowledgements and references for MedSAM provided by Ma et al. are retained at the end of this README, and the original license for conditions of redistributing their work is available at https://github.com/bowang-lab/MedSAM?tab=Apache-2.0-1-ov-file.

In this derivative work, we made the following modifications:
1. Used `matplotlib` instead of `skimage.io` for saving plots in `MedSAM_Inference.py`
2. Copied and modified our own versions of `train_one_gpu.py` (now `main.py` in root dir) and  `train_multi_gpus.pypi` (now `main_ddp.py` in root dir)
* `main.py` and `main_ddp.py` were re-reorganized such that all function definitions are at the beginning of the scripts.
* The parser setup was moved to under the `if` statement, now after all function definitions. 
* `HyperOptArgumentParser` was used instead of `ArgumentParser` due to the added ease of reading from a `.json` file. 
3. The `MedSAM` model and custom Dataset (`SegMRIDataset` in place of `NpyDataset`) classes were moved to their own `.py` files and imported into `main.py` and `main_ddp.py` for better readability, modularity, and personal coding preference. 
4. We aim to add Optuna functionality to `main.py` and `main_ddp.py` for automated hyperparameter optimization.

To install MedSAM, clone the original MedSAM repo at https://github.com/bowang-lab/MedSAM and follow the authors' installation instructions (copied below as well for ease of access). 


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Get Started
Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at e.g., `work_dir/MedSAM/medsam_vit_b`

We provide three ways to quickly test the model on your images

1. Command line

```bash
python MedSAM_Inference.py # segment the demo image
```

Segment other images with the following flags
```bash
-i input_img
-o output path
--box bounding box of the segmentation target
```

2. Jupyter-notebook

We provide a step-by-step tutorial on [CoLab](https://colab.research.google.com/drive/19WNtRMbpsxeqimBlmJwtd1dzpaIvK2FZ?usp=sharing)

You can also run it locally with `tutorial_quickstart.ipynb`.

3. GUI

Install `PyQt5` with [pip](https://pypi.org/project/PyQt5/): `pip install PyQt5 ` or [conda](https://anaconda.org/anaconda/pyqt): `conda install -c anaconda pyqt`

```bash
python gui.py
```


## Model Training

### Data preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip it to `data/FLARE22Train/`.

This dataset contains 50 abdomen CT scans and each scan contains an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).

Run pre-processing

Install `cc3d`: `pip install connected-components-3d`

```bash
python pre_CT_MR.py
```

- split dataset: 80% for training and 20% for testing
- adjust CT scans to [soft tissue](https://radiopaedia.org/articles/windowing-ct) window level (40) and width (400)
- max-min normalization
- resample image size to `1024x2014`
- save the pre-processed images and labels as `npy` files


### Training on multiple GPUs (Recommend)

The model was trained on five A100 nodes and each node has four GPUs (80G) (20 A100 GPUs in total). Please use the slurm script to start the training process.

```bash
sbatch train_multi_gpus.sh
```

When the training process is done, please convert the checkpoint to SAM's format for convenient inference.

```bash
python utils/ckpt_convert.py # Please set the corresponding checkpoint path first
```

### Training on one GPU

```bash
python train_one_gpu.py
```

If you only want to train the mask decoder, please check the tutorial on the [0.1 branch](https://github.com/bowang-lab/MedSAM/tree/0.1).


## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


## Reference

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={1--9},
  year={2024}
}
```
