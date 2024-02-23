#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:07:14 2017

@author: ldy
"""

from __future__ import print_function
from os.path import exists, join, basename
from os import makedirs, remove
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from model import LasSRN
import os
import torch.nn as nn

rcParams['figure.figsize'] = 40, 24
rcParams.update({'font.size': 22})
# Training settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--test_folder', type=str, default='./dataset/BSDS300/images/train', help='input image to use')
parser.add_argument('--model', type=str, default='model/model_epoch_10.pth', help='model file to use')
parser.add_argument('--save_folfer', type=str, default='./results', help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

# 裁剪图像使其宽高是8的倍数
def centeredCrop(img):
    width, height = img.size   # Get dimensions
    new_width = width - width % 8
    new_height = height - height % 8 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return img.crop((left, top, right, bottom))

# 将模型输出的Y通道图像与原始的Cb和Cr通道合并，然后转换回RGB色域。
def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    # Resize cb and cr to match the size of out_img_y using PIL's Image.resize() method
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

    # Merge Y, Cb, and Cr channels back to an image
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img


def save_image(HR_2, HR_4, HR_8, GT, name):
    LR = y.resize((y.size[0] // 8, y.size[1] // 8), Image.BICUBIC)
    Bicubic_HR_2 = LR.resize((y.size[0]*2, y.size[1]*2), Image.BICUBIC)
    Bicubic_HR_4 = LR.resize((y.size[0]*4, y.size[1]*4), Image.BICUBIC)
    Bicubic_HR_8 = LR.resize((y.size[0]*8, y.size[1]*8), Image.BICUBIC)

    fig = plt.figure()
    
    ax = plt.subplot(251)
    ax.axis("off")
    ax.imshow(LR)
    ax.set_title("LR")
    
    ax = plt.subplot(252)
    ax.axis("off")
    ax.imshow(HR_2)
    ax.set_title("LapSAN HR_2")
    
    ax = plt.subplot(253)
    ax.axis("off")
    ax.imshow(HR_4)
    ax.set_title("LapSAN HR_4")
    
    ax = plt.subplot(254)
    ax.axis("off")
    ax.imshow(HR_8)
    ax.set_title("LapSAN HR_8")
    
    ax = plt.subplot(255)
    ax.axis("off")
    ax.imshow(GT)
    ax.set_title("GT")

    ax = plt.subplot(256)
    ax.axis("off")
    ax.imshow(LR)
    ax.set_title("LR")
    
    ax = plt.subplot(257)
    ax.axis("off")
    ax.imshow(Bicubic_HR_2)
    ax.set_title("Bicubic HR_2")
    
    ax = plt.subplot(258)
    ax.axis("off")
    ax.imshow(Bicubic_HR_4)
    ax.set_title("Bicubic HR_4")
    
    ax = plt.subplot(259)
    ax.axis("off")
    ax.imshow(Bicubic_HR_8)
    ax.set_title("Bicubic HR_8")
    
    ax = plt.subplot(2,5,10)
    ax.axis("off")
    ax.imshow(GT)
    ax.set_title("GT")

    # Save images
    # 使用os来构建文件路径，确保兼容性
    if not os.path.exists(opt.save_folfer):
        os.makedirs(opt.save_folfer)
    fig.savefig(os.path.join(opt.save_folfer+'/'+name+'.png'))
    print ('image:'+name+'saved!')

images_list = glob(opt.test_folder+'/*.jpg')
print (len(images_list))

# model = torch.load(opt.model)
#加载模型状态字典
model_dict = torch.load(opt.model)
#创建一个新的神经网络模型对象
model = LasSRN()
model.load_state_dict(model_dict)

if opt.cuda:
    model = model.cuda()
for image_path in images_list:
    img_name_with_extension = os.path.basename(image_path)
    img_name = os.path.splitext(img_name_with_extension)[0]
    # img_name = image_path.split('/')[-1].split('.')[0]
    img = Image.open(image_path).convert('YCbCr')
    img = centeredCrop(img)
    y, cb, cr = img.split()
    LR = y.resize((y.size[0]//8, y.size[1]//8), Image.BICUBIC)
    print (LR.size)
    LR = ToTensor()(LR).unsqueeze(0)  # Add batch dimension
    if opt.cuda:
        LR = LR.cuda()
    HR_2, HR_4, HR_8 = model(LR)
    HR_2 = HR_2.cpu()
    HR_4 = HR_4.cpu()
    HR_8 = HR_8.cpu()
    HR_2 = process(HR_2, cb, cr)
    HR_4 = process(HR_4, cb, cr)
    HR_8 = process(HR_8, cb, cr)
    img = img.convert("RGB")
    save_image(HR_2, HR_4, HR_8, img, img_name)
    

