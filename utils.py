# --- Imports --- #
import math
from math import exp
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
import numpy as np
from torch.autograd import Variable
# import skimage.measure
# from skimage.metrics import structural_similarity as SSIM   
# from skimage.metrics import peak_signal_noise_ratio as PSNR
# --- Commands ---#
# CUDA_VISIBLE_DEVICES=0 python3 test.py

# def PSNR(img1, img2):
#     mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def to_psnr(pred_image, gt):
    
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


# def to_ssim_skimage(pred_image, gt):
#     pred_image_list = torch.split(pred_image, 1, dim=0)
#     gt_list = torch.split(gt, 1, dim=0)

#     pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
#     gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
#     # ssim_list = [skimage.measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]
#     ssim_list = [compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]
#     return ssim_list


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def PSNR(img1, img2):
    # print(img1.unique()) # ([0.2730, 0.2820, 0.2837,  ..., 0.6364, 0.6400, 0.6514])
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    # mse = np.mean((np.array(img1) / 255. - np.array(img2) / 255.) ** 2)
    # print("PSNR:", mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

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


def validation(net, val_data_loader, device, category, exp_name, save_tag=False):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad(): # release gpu
            input_im, gt, image_name = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            # print(input_im.shape) # [1, 3, 1024, 464]
            pred_image,zy_in = net(input_im)
            # print(pred_image.shape, zy_in.shape) # [1, 3, 1024, 464] [1, 32, 128, 58]
            # print(torch.max(pred_image)) # 0.8816
        # --- Calculate the average PSNR --- #
        print("extend:", to_psnr(pred_image.cpu(), gt.cpu()))
        print("append:", PSNR(pred_image.cpu(), gt.cpu()))
        psnr_list.extend(to_psnr(pred_image.cpu(), gt.cpu()))
        # psnr_list.append(PSNR(pred_image.cpu(), gt.cpu()))
        # --- Calculate the average SSIM --- #
        ssim_list.append(SSIM(pred_image.cpu(), gt.cpu()))
        #print(image_name,psnr_list[-1],ssim_list[-1])
        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, category, exp_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        utils.save_image(pred_image_images[ind], './{}_results/{}/{}'.format(category, exp_name, image_name_1[:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_{}_log.txt'.format(category, exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)

def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 100 if category == 'derain' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# # 初始化
# ema = EMA(model, 0.999)
# ema.register()

# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()

# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()