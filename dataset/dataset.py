from torch.utils.data import Dataset
# import cv2 as cv
import numpy as np
import torch
from os import path, listdir

class BaseDataset(Dataset):
    def __init__(self,
                 root_path=None,
                 splits_path=None,
                 split='train',
                 split_extension='txt',
                 split_separator=' ',
                 split_skiplines=0,
                 resize_to=None,
                 crop_to=None,
                 augment_data=True,
                 sensors=['rgb'],
                 return_grayscale=False,
                 depth_mode='log',
                 **kwargs): # whether to use city19 or city36 class set

        self.root_path = root_path
        self.sensors = sensors
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.kwargs = kwargs
        self.augment_data = augment_data
        self.return_grayscale = return_grayscale
        self.depth_mode = depth_mode

        with open(path.join(splits_path, split+'.'+split_extension)) as f:
            #self.items = [l.rstrip('\n').split(split_separator) for l in f][split_skiplines:]
            # This line of code reads a text file containing image paths and labels, and creates a
            # list of items where each item is a list containing the image path and label. The
            # `split_separator` argument is used to split each line of the text file into separate
            # elements, and the `split_skiplines` argument is used to skip a certain number of lines
            # at the beginning of the file. The `e.lstrip('/')` method is used to remove any leading
            # forward slashes from the image path. The resulting list of items is stored in the
            # `self.items` attribute of the dataset object.
            self.items = [[e.lstrip('/') for e in l.rstrip('\n').split(split_separator)] for l in f][split_skiplines:]

        self.init_ids()
        self.init_cmap()
        self.init_cnames()

    # to be overridden
    def init_ids(self):
        self.raw_to_train = {i:i for i in range(256)}
        self.ignore_index = -1

    # to be overridden
    def init_cmap(self):
        self.cmap = np.array([[i,i,i] for i in range(256)])
        
    def init_cnames(self):
        self.cnames = ["c%03d"%i for i in range(256)]

    def __getitem__(self, item):
        rgb_path, gt_path = self.items[item]

        rgb = self.load_rgb(path.join(self.root_path, rgb_path)) if 'rgb' in self.sensors else None
        gt = self.map_to_train(self.load_semantic(path.join(self.root_path, gt_path))) if 'semantic' in self.sensors else None

        rgb, gt, _ = self.resize_and_crop(rgb=rgb, gt=gt)
        if self.augment_data:
            rgb, gt, _ = self.data_augment(rgb=rgb, gt=gt)
        rgb, gt, _ = self.to_pytorch(rgb=rgb, gt=gt)

        out_dict = {}
        if rgb is not None: out_dict['rgb'] = rgb
        if gt is not None: out_dict['semantic'] = gt
        #if depth is not None: out_dict['depth'] = depth

        return out_dict, item

    def __len__(self):
        return len(self.items)

    def to_pytorch(self, rgb=None, gt=None, depth=None):
        if not self.return_grayscale:
            if rgb is not None:
                rgb = torch.from_numpy(np.transpose((rgb[...,::-1]-[104.00698793, 116.66876762, 122.67891434]), (2, 0, 1)).astype(np.float32))
                #rgb = torch.from_numpy(np.transpose((rgb[...,::-1]/255.-[0.485, 0.456, 0.406])/[0.485, 0.456, 0.406], (2, 0, 1)).astype(np.float32))
        else:
            if rgb is not None:
                rgb = torch.from_numpy(np.transpose(np.expand_dims(cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)/127.5 -1, -1), (2, 0, 1)).astype(np.float32))
        if gt is not None:
            gt = torch.from_numpy(gt.astype(np.long))
        if depth is not None:
            if self.depth_mode == 'linear':
                depth = torch.from_numpy((2*depth-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            elif self.depth_mode == 'log':
                depth = torch.from_numpy((2*(np.log2(depth+1.))-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            elif self.depth_mode == 'root4':
                depth = torch.from_numpy((2*np.power(depth, 1/4)-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
            else:
                depth = torch.from_numpy((2*np.sqrt(depth)-1.).astype(np.float32)).unsqueeze(0) # depth should be normalized in [0,1] before input
        return rgb, gt, depth

    def to_rgb(self, tensor, force_gray=False):
        if not (self.return_grayscale or force_gray):
            t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.67891434]
        else:
            t = (np.array(tensor.transpose(0,1).transpose(1,2))+1.)*127.5
        t = np.round(t).astype(np.uint8) # rgb
        return t

    def color_label(self, gt):
        return self.cmap[np.array(gt)]

    def map_to_train(self, gt):
        gt_clone = self.ignore_index*np.ones(gt.shape, dtype=np.long)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                gt_clone[gt==k] = v
        return gt_clone
