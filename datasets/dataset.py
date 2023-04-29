import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from datasets.zdataset import mvsecdataset, forthdataset




class MvsecDataset(Dataset):
    def __init__(self, data_split, datapath, list_filename, training):
        self.data_split = str(data_split)
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
        
        if not os.path.exists(self.datapath+'/frameindoors'+self.data_split+'_mvsec3'):
            os.makedirs(self.datapath+'/frameindoors'+self.data_split+'_mvsec3')
            mvsecdataset(self.datapath, self.data_split, True)
            
        self.datapath = self.datapath+'/frameindoors'+self.data_split+'_mvsec3'

   
    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        # return Image.open(filename).convert('RGB')
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data



    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
       
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

    

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w = 346
            h = 260
            crop_w, crop_h = 320,256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)


            # random crop
            left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w,:]
            right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w,:]
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img/255.0)
            right_img = processed(right_img/255.0)
            


            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w = 346
            h = 260


            # normalize
            processed = get_transform()
            left_img = processed(left_img/255.0).numpy()
            right_img = processed(right_img/255.0).numpy()

            # pad to size 384*288
            top_pad = 288 - h
            right_pad = 384 - w
            assert top_pad > 0 and right_pad > 0
            
            
            left_img = np.lib.pad(left_img, ((0, 0),(top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0),(top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            #pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            
            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}



class ForthDataset(Dataset):
    def __init__(self, data_filename, datapath, list_filename, training):
        self.data_filename = data_filename
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
        
        if not os.path.exists(self.datapath+'/frameindoor_forth'):
            os.makedirs(self.datapath+'/frameindoor_forth')
            forthdataset(self.datapath, self.data_filename, True)
            
        self.datapath = self.datapath+'/frameindoor_forth'

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images        
        
    def load_image(self, filename):
        # return Image.open(filename).convert('RGB')
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))



        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:

            w = 640
            h = 480
            crop_w, crop_h = 320,256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)


            # random crop

            left_img = left_img[y1:y1 + crop_h, x1:x1 + crop_w,:]
            right_img = right_img[y1:y1 + crop_h, x1:x1 + crop_w,:]
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]


            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img/255.0)
            right_img = processed(right_img/255.0)
            
    

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:

            w = 640
            h = 480
      

            # normalize
            processed = get_transform()
            left_img = processed(left_img/255.0).numpy()
            right_img = processed(right_img/255.0).numpy()

            # pad to size 704*640
            top_pad = 640 - h
            right_pad = 704 - w
            assert top_pad > 0 and right_pad > 0
            
            left_img = np.lib.pad(left_img, ((0, 0),(top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0),(top_pad, 0), (0, right_pad)), mode='constant',
                                    constant_values=0)
         
            #pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            
            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}


            

