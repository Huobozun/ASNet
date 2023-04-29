from pathlib import Path
import numpy as np
import struct
import cv2 as cv
import torch
import re
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('tkaGg')  # 大小写无所谓 tkaGg ,TkAgg 都行

 
 
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

gt_folder = '/home/wpf/zjg/ASNet/frameindoor_mvsec3/depth'
gtp_folder = '/home/wpf/zjg/ASNet/frameindoor_mvsec3/predictions'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for iee in range(0,1874):
    


    disp_ground, scale0 = pfm_imread('{}/{}.pfm'.format(gt_folder,iee))
    disp_est, scale1 = pfm_imread('{}/{}.pfm'.format(gtp_folder,iee))


    
    #MDE
    disp_ground = np.array(disp_ground)
    disp_est = np.array(disp_est)


    mask = (disp_ground < 10) & (disp_ground > 0)
    # mask = (disp_ground > 0)
    disp_est[mask == False] = 0
    disp_ground[mask == False] = 0
    
    plt.imsave('/home/wpf/zjg/ASNet/frameindoor_mvsec3/depthshow/{}.png'.format(iee),disp_ground)
    plt.imsave('/home/wpf/zjg/ASNet/frameindoor_mvsec3/predictionshow/{}.png'.format(iee),disp_est)
    
    
   










