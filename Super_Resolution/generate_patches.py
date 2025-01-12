## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

##### Data preparation file for training MIRNetv2 on the RealSR Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=str, required=True, help='x2,x3,x4')
parser.add_argument('--finetune', type = str, required = True, help = 'False, True')
args = parser.parse_args()



def train_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename = os.path.join(hr_tar, filename + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)


############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0

first_hr_path = '/content/lte-CTscan/load/CT/CT_train_HR'
first_sr_path = '/content/lte-CTscan/load/CT/CT_train_LR_bilinear/'+args.scale

finetune_hr_path = '/content/CT/CT_train_HR'
finetune_sr_path = '/content/CT/CT_train_LR_bilinear/' + args.scale

# src_hr = '/content/lte-CTscan/load/CT/CT_train_HR'
# src_lr = '/content/lte-CTscan/load/CT/CT_train_LR_bilinear/'+args.scale


src_hr = finetune_hr_path if args.finetune == 'True' else first_hr_path
src_lr = finetune_sr_path if args.finetune == 'True' else first_sr_path

tar = 'Datasets/train/CT/'+args.scale

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(os.listdir(src_lr))
hr_files = natsorted(os.listdir(src_hr))

files = [(os.path.join(src_lr,i), os.path.join(src_hr,j)) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))

