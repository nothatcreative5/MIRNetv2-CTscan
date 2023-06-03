## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/


import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/content/MIRNetv2/basicsr')

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from utils import MIRNet_v2
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Super-Resolution using MIRNet-v2')

parser.add_argument('--input_dir', default='/content/CT/CT_test_LR_bilinear', type=str, help='Directory of test images')
parser.add_argument('--gt_dir', default='/content/CT/CT_test_HR', type=str, help='Directory of ground truth images')
parser.add_argument('--result_dir', default='./results/CT/', type=str, help='Directory for results')
parser.add_argument('--scale', default='X4', type=str, help='Scale factor for super-resolution')

args = parser.parse_args()


####### Load yaml #######
if args.scale=='X2':
    yaml_file = '/content/MIRNetv2/Super_Resolution/Options/SuperResolution_MIRNet_v2_scale2_finetune.yml'
    weights = '/content/MIRNetv2/experiments/SuperResolution_MIRNet_v2_scale2/models/net_g_latest.pth'
elif args.scale=='X3':
    yaml_file = '/content/MIRNetv2/Super_Resolution/Options/SuperResolution_MIRNet_v2_scale3_finetune.yml'
    weights = '/content/MIRNetv2/experiments/SuperResolution_MIRNet_v2_scale3/models/net_g_latest.pth'
elif args.scale=='X4':
    yaml_file = '/content/MIRNetv2/Super_Resolution/Options/SuperResolution_MIRNet_v2_scale4_finetune.yml'
    weights = '/content/MIRNetv2/experiments/SuperResolution_MIRNet_v2_scale4/models/net_g_latest.pth'
else: 
    print("Wrong SR scaling factor")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = MIRNet_v2(**x['network_g'])

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


img_multiple_of = 4
scale = args.scale
result_dir  = os.path.join(args.result_dir, scale)
os.makedirs(result_dir, exist_ok=True)

input_dir = os.path.join(args.input_dir, scale)
input_paths = natsorted(os.listdir(input_dir + '/'))
input_gt_paths = natsorted(os.listdir(args.gt_dir + '/'))

ssims = []
psnrs = []


with torch.inference_mode():
    for inp_path, gt_path in tqdm(zip(input_paths, input_gt_paths), total=len(input_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


        inp_path = os.path.join(input_dir, inp_path)
        gt_path = os.path.join(args.gt_dir, gt_path)

        img = np.float32(utils.load_img(inp_path))/255.

        gt = np.float32(utils.load_img(gt_path))/255.

        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr = utils.calculate_psnr(gt * 255.0, restored * 255.0)
        ssim = utils.calculate_ssim(gt * 255.0, restored * 255.0)

        psnrs.append(psnr)
        ssims.append(ssim)

        print("PSNR: {:.2f} dB SSIM: {:.4f}".format(psnr, ssim))

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))

print("Avg PSNR: {:.2f} dB Avg SSIM: {:.4f}".format(np.mean(psnrs), np.mean(ssims)))