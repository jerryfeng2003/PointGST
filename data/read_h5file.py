import h5py
import os
import sys
import glob
import h5py
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.logger import *
import torch

ckpt_path = "/data/xzhou/xyh/checkpoint/pretrain/mae/pretrain.pth"
state_dict = torch.load(ckpt_path, map_location='cpu')
print(state_dict.keys())
base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
print(base_ckpt.keys())