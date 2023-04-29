import re
import numpy as np
import torchvision.transforms as transforms


def get_transform():
    # mean = [0.485, 0.456, 0.406]
    #0 126.6586722946167 21.980125 4.688296
    #1 131.76563822428386 60.921598 7.805229
    #2 130.4244837697352 59.627553 7.721888
    # mean = [0.4967, 0.5167, 0.5115]
    # std = [0.01878, 0.03168, 0.03126]
    mean = [128.0/255.0, 128.0/255.0, 128.0/255.0]
    std = [0.018386, 0.030609, 0.030282]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )



def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


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
