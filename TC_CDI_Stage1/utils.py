
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn



def generate_masks_test(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x


def split_feature(x):
    l = x.shape[1]
    x1 = x[:, 0:l // 2, ::]
    x2 = x[:, l // 2:, ::]
    return x1, x2

class rev_3d_part(nn.Module):

    def __init__(self, in_ch):
        super(rev_3d_part, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )
        self.g1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = split_feature(x)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.g1(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def reverse(self, y):
        y1, y2 = split_feature(y)
        x2 = y2 - self.g1(y1)
        x1 = y1 - self.f1(x2)
        x = torch.cat([x1, x2], dim=1)
        return x