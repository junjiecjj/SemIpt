# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import sys,os
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

import glob
import random
import pickle
import io

import PIL.Image as pil_image
from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import torchvision.transforms as tfs

# 本项目自己编写的库
from option import args





