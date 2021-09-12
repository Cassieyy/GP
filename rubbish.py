import pdb
import math
import torch
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from skimage.measure import compare_psnr
import math
from math import exp

# from skimage.metrics import structural_similarity as SSIM   
# from skimage.metrics import peak_signal_noise_ratio as PSNR
# --- Commands ---#
# CUDA_VISIBLE_DEVICES=0 python3 test.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def compute_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def to_psnr(pred_image, gt):
    
    mse = F.mse_loss(pred_image, gt, reduction='none')
    # print("to_psnr:", mse.shape)
    # assert 1>3
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    
    intensity_max = 1.0
    psnr_list = [10.0 * math.log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def PSNR(img1, img2):
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    # mse = np.mean((np.array(img1) / 255. - np.array(img2) / 255.) ** 2)
    # print("PSNR:", mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

if __name__ == "__main__":
    
    
    # input1 = np.random.randn(256, 256, 3)
    # input2 = np.random.randn(256, 256, 3)
    input1 = torch.randn(1, 3, 16, 16).cuda(1)
    print(input1.requires_grad)
    net = nn.Conv2d(3, 1, 3, 1, 1).cuda(1)
    output1 = net(input1)
    print(output1.shape)
    print(output1.requires_grad)

    assert 1>3
    input1 = torch.nn.Parameter(input1)
    print(input1.requires_grad)
    input1 = input1.detach()
    print(input1.requires_grad)
    assert 1>3
    input2 = torch.randn(1, 3, 256, 256).cuda()
    # a = to_psnr(input1, input2)
    a = PSNR(input1.cpu(), input2.cpu())
    print(a)
    assert 1>3
    a = SSIM(input1, input1)
    # print(a)
    # psnr2 = compute_PSNR(input1, input2)
    # ssim = SSIM(pred, gt_image, multichannel=True)
    # out = F.upsample_nearest(inputs, scale_factor=2)
    # psnr = compare_psnr(np.array(input1), np.array(input2), 255)
    # print("psnr:", psnr)
    psnr1 = PSNR(input1, input1)
    print(psnr1)
    # psnr2 = to_psnr(input1, input2)
    # print(psnr, psnr1, psnr2)
    # assert 1>3
    # out = F.interpolate(inputs, scale_factor=2)
    # print(out.shape)

    
