from pathlib import Path
import numpy as np
import struct
import cv2 as cv
import torch
import re
import torch.nn.functional as F
import matplotlib
from models import __models__, model_loss
import matplotlib.pyplot as plt



 
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  
        endian = '<'
        scale = -scale
    else:
        endian = '>'  

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

gt_folder = '/media/wpf/hdd/zjg/mvsec/frameindoors1_mvsec3/TRAIN'
gtp_folder = '/home/wpf/zjg/ASnet/predictions_new2depths1_TRAIN'

all_mde= 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for iee in range(0,3114):
    
# for iee in range(0,2):




    disp_ground, scale0 = pfm_imread('{}/depthleft/{}.pfm'.format(gt_folder,iee))
    disp_est, scale1 = pfm_imread('{}/{}.pfm'.format(gtp_folder,iee))


    
    #MDE
    disp_ground = np.array(disp_ground)
    disp_est = np.array(disp_est)
    # print(disp_ground)
    # print(disp_est)



    # mask = (disp_ground < 4) & (disp_ground > 3)
    mask = (disp_ground > 4)
    res = disp_est - disp_ground
    res[mask == False] = 0
    
    rcount = disp_est
    rcount[mask == False] = 0
    rcount[mask == True] = 1
    ma = np.sum(abs(res))
    mb = np.sum(abs(rcount))
    if(mb ==0):
        continue
    mde = ma/mb

        

    print(2,iee,all_mde,mde)
    all_mde=(all_mde*iee+mde)/(iee+1)
    iee+=1
    # print(2,error_rate)


    print(all_mde)
   
#1TRAIN 0-1:0.15953 1-2:0.09027 2-3:0.12061 3-4:0.14921 >4:0.24822

#1 train 0.1145 TEST 0.2046
#2 train 0.1017 TEST 0.2874
#3 train 0.1322 TEST 0.2215












