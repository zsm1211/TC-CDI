

from utils import generate_masks_test, time2file_name
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
from skimage.metrics import structural_similarity as SSIM
import cv2
from models import*


if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_path = r'./mask'

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--last_train', default=34, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='model', type=str, help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--num_block', default=26, type=int, help='number of reversible blocks')
parser.add_argument('--B', default=20, type=int, help='compressive rate')
parser.add_argument('--learning_rate', default=0.00005, type=float)
parser.add_argument('--size', default=[512, 512], type=int, help='input image resolution')
parser.add_argument('--mode', default='noreverse', type=str, help='training mode: reverse or noreverse')
parser.add_argument('--add_noise_training', default=False, type=bool, help='add synthetic noise')

args = parser.parse_args()
mask, mask_s = generate_masks_test(mask_path)


loss = nn.MSELoss()
loss.cuda()

def test_for_real(model):
    meas = scio.loadmat(r'./meas_all.mat')['meas']
    meas = torch.from_numpy(meas).cuda().float()
    meas_re = torch.div(meas, mask_s)
    meas_re = torch.unsqueeze(meas_re, 1)
    maskt = mask.expand([1, args.B, args.size[0], args.size[1]])
    Phi = maskt.cuda().float()
    Phi_s = torch.sum(Phi, 1)
    Phi_s[Phi_s == 0] = 1
    out_save1 = torch.zeros([meas.shape[0], args.B, args.size[0], args.size[1]]).cuda()
    for i in range(1):#meas.shape[0]
        with torch.no_grad():
            print(i)
            out_pic1 = model(meas[i:i+1, :, :], Phi, Phi_s, meas_re, args)[-1]
            torch.cuda.synchronize()
            out_save1[i, :, :, :] = out_pic1[0, :, :, :]
    for i in range(1):#meas.shape[0]
        for j in range(20):
            cv2.imwrite(r'./result/{}_{}.png'.format('result', j + 20 * i),out_save1.detach().cpu().numpy()[i, j, :, :] * 255)


if __name__ == '__main__':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_model=GAP_net(args).cuda()
    model_parameter = torch.load(r'./model/base/model_state_epoch_1_it_4950.pth')
    test_model.load_state_dict(model_parameter,strict=False)

    test_for_real(test_model.eval())

