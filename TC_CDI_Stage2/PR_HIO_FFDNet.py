import time
import numpy as np
import imageio
import cv2
from packages.ffdnet.models import FFDNet
import math
import torch
import scipy.io as sio
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser

net = FFDNet(num_input_channels=1).cuda()
model_fn = 'packages/ffdnet/models/net_gray.pth'
state_dict = torch.load(model_fn)
net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.load_state_dict(state_dict)

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

magnitudes_oversampled =imageio.imread(r'./meas.bmp')[0:800,:800]
# magnitudes_oversampled=cv2.medianBlur(magnitudes_oversampled,3)#median filter (optional)
magnitudes_oversampled=np.pad(magnitudes_oversampled, 200, 'constant')




def LF_NNPsupport(g1, percent):
    g=np.abs(g1)
    gs=np.sort(g.reshape(-1))
    (m,n)=g.shape
    thre=gs[int(np.round(m*n*(1-percent)))]
    S = (g >= thre)
    Num=np.sum(S!=0)
    AVRG=np.sum(S*g)/Num
    g2=g1-0.4*g1*(g>(4*AVRG))
    return S,g2


def LNNPPhaseRetrieval(sautocorr,beta_start,beta_step,beta_stop,N_iter,init_guess,percent):
    g1=init_guess
    ii=0
    cor=[]
    BETAS = np.array(range(beta_start, beta_stop, beta_step))/100
    for ibeta in range(len(BETAS)):
        beta = BETAS[ibeta]
        for i in range(N_iter):
            ii=ii+1
            S, g1 = LF_NNPsupport(g1, percent)
            G_uv = np.fft.fft2(g1)
            g1_tag =np.real(np.fft.ifft2(sautocorr*G_uv/np.abs(G_uv)))
            g1=g1_tag*(g1_tag>=0)*S+(g1 - beta*g1_tag)*(g1_tag<0)*S+ (g1 - beta*g1_tag)*(1-S)
            if i < 15:
                g1 = g1 * (g1_tag >= 0) * S
                Max = np.max(g1)
                g1 = ffdnet_vdenoiser(g1 / Max, 5 / 255, net) * Max #tunable parameter of noise level for different input; uncomment for FFDNet
            cor.append(corr2(abs(G_uv),sautocorr))
    for iter in range(N_iter):
        ii = ii + 1
        G_uv = np.fft.fft2(g1)
        g1_tag = np.real(np.fft.ifft2(sautocorr * G_uv / np.abs(G_uv)))
        if iter<15:
            g1 = g1 * (g1_tag >= 0) * S
            Max = np.max(g1)
            g1 = ffdnet_vdenoiser(g1 / Max, 5 / 255, net) * Max #tunable parameter of noise level for different input; uncomment for FFDNet
        cor.append(corr2(abs(G_uv),sautocorr))
    recons_err = np.mean(np.power((np.abs(np.fft.fft2(g1)) - sautocorr),2))
    recons_err2 = np.sqrt(np.mean(np.power((np.power(np.abs(np.fft.fft2(g1)),2)-np.power(sautocorr,2)),2)))
    return g1,cor,recons_err,recons_err2

trial=100
percent_list=[0.001,0.002,0.003,0.004,0.005,0.006]#tunable list for different input
for k in range(trial):
    steps=60
    for percent in percent_list:
        start=time.time()
        Reconstruct_Field,a,b,c=LNNPPhaseRetrieval(np.fft.ifftshift(magnitudes_oversampled),40,-4,0,steps,np.random.rand(*magnitudes_oversampled.shape)+1j*np.random.rand(*magnitudes_oversampled.shape),percent)
        Reconstruct_Image_Inten = np.power(np.abs(Reconstruct_Field),2)
        Reconstruct_Image_Amp = np.abs(Reconstruct_Field)
        end=time.time()
        print("Saving_{}_time:{}s".format(k,end-start))
        sio.savemat(r'.\result\New_result_num={}_percent={}.mat'.format(k, percent),{'result': Reconstruct_Field})
        cv2.imwrite(r'.\result\New_FFD_ML_result_percent={}_k={}.png'.format(percent,k),Reconstruct_Image_Amp/np.max(Reconstruct_Image_Amp)*255)