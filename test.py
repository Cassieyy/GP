
# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from model import  DeRain_v2
from utils import validation
import os
import numpy as np
import random
# CUDA_VISIBLE_DEVICES=1 python3 test.py -lambda_loss 0.0015

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='DDN_SIRR_withGP', type=str)
parser.add_argument('-category', help='Set image category (derain or dehaze?)', default='deblur', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
exp_name = args.exp_name

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size,lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'deblur':
    val_data_dir = './data/test/deblur/'
elif category == 'dehaze':
    val_data_dir = './data/test/dehaze/'
else:
    raise Exception('Wrong image category. Set it to derain or dehaze dateset.')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_filename = 'test_list.txt'
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = DeRain_v2()

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('./{}/{}_best.pth'.format(exp_name,category)))
# net.load_state_dict(torch.load('./{}/{}.pth'.format(exp_name,category)))


# --- Use the evaluation model in testing --- #
net.eval()
if os.path.exists('./{}_results/{}/'.format(category,exp_name))==False: 	
	os.makedirs('./{}_results/{}/'.format(category,exp_name))	
	os.makedirs('./{}_results/{}/blur/'.format(category,exp_name))
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net, val_data_loader, device, category, exp_name, save_tag=True)
end_time = time.time() - start_time
print('test_psnr: {0:.2f}, test_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('inference time is {0:.4f}'.format(end_time))
