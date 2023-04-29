from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
from utils.KittiColormap import *
from datasets.data_prepare import dataprepare
import time


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='ASNet')
parser.add_argument('--model', default='ASNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdepth', type=int, default=10, help='maximum depth')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datasplit', help='data spilt',type=int, default=1)
parser.add_argument('--datafilename', help='data file name',type=str, default='event7.aedat4')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')


# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]


if(args.dataset == 'mvsec'):
    if not os.path.exists('./datasets/prepare.txt'):
        dataprepare(args.dataset,args.datapath)
    test_dataset = StereoDataset(args.datasplit,args.datapath, args.testlist, False)
    
else:
    if not os.path.exists('./datasets/prepare4.txt'):
        dataprepare(args.dataset,args.datapath+'/'+args.datafilename)
    test_dataset = StereoDataset(args.datafilename,args.datapath, args.testlist, False)

         
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=True)



# model, optimizer
model = __models__[args.model](args.maxdepth)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)
    
    start=time.time()

    for batch_idx, sample in enumerate(TestImgLoader):
        

        disp_est_tn = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_tn)
        disp_est_np = np.array([disp_est_np])
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        

        

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            
            disp_est = disp_est[0]
            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            name = fn.split('/')
            fnplt = os.path.join("predictions", 'plt_'+name[-1])
            fnplt = fnplt.replace('pfm','jpg')

            fn = os.path.join("predictions", name[-1])

            




            print(fn)
            cv2.imwrite(fn, disp_est)
            plt.imsave(fnplt, disp_est)

 
                     
    end =time.time()
    

    print("Done!")
    print("time:",end-start)

def save_txt(path,data):
	with open(path, 'w') as outfile:
		for slice_2d in data:
			np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]

DISPARITY_MULTIPLIER = 7.0
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
}

def depth_to_disparity(depth_maps):
    """
    Conversion from depth to disparity used in the paper "Learning an event sequence embedding for dense event-based
    deep stereo" (ICCV 2019)

    Original code available at https://github.com/tlkvstepan/event_stereo_ICCV2019
    """
    disparity_maps = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (depth_maps + 1e-15)
    return disparity_maps


def disparity_to_depth(disparity_map):
    depth_map = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (disparity_map + 1e-7)
    return depth_map


if __name__ == '__main__':
    test(args)
