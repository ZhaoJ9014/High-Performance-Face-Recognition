import collections
import os.path as osp
# from __future__ import division

import numpy as np
import PIL.Image
import scipy.io
import skimage
import skimage.color as color
from skimage.transform import rescale
from skimage.transform import resize
import torch
from torch.utils import data



DEBUG = False



class DemoFaceDataset(data.Dataset):
    '''
        Dataset subclass for demonstrating how to load images in PyTorch.

    '''

    # -----------------------------------------------------------------------------
    def __init__(self, root, split='train', set='tiny', im_size=250):
    # -----------------------------------------------------------------------------
        '''
            Parameters
            ----------
            root        -   Path to root of ImageNet dataset
            split       -   Either 'train' or 'val'
            set         -   Can be 'full', 'small' or 'tiny' (5 images)
        ''' 
        self.root = root  # E.g. '.../ImageNet/images' or '.../vgg-face/images'
        self.split = split
        self.files = collections.defaultdict(list)
        self.im_size = im_size # scale image to im_size x im_size
        self.set = set

        if set == 'small':
            raise NotImplementedError()
            
        elif set == 'tiny':
            # DEBUG: 5 images
            files_list = osp.join(root, 'tiny_face_' + self.split + '.txt')

        elif set == 'full':
            raise NotImplementedError()

        else:
        	raise ValueError('Valid sets: `full`, `small`, `tiny`.')

        assert osp.exists(files_list), 'File does not exist: %s' % files_list

        imfn = []
        with open(files_list, 'r') as ftrain:
            for line in ftrain:
                imfn.append(osp.join(root, line.strip()))
        self.files[split] =  imfn


    # -----------------------------------------------------------------------------
    def __len__(self):
    # -----------------------------------------------------------------------------
        return len(self.files[self.split])


    # -----------------------------------------------------------------------------
    def __getitem__(self, index):
    # -----------------------------------------------------------------------------
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)

        # HACK: for non-RGB images - 4-channel CMYK or 1-channel grayscale
        if len(img.getbands()) != 3:
            while len(img.getbands()) != 3:
                index -= 1
                img_file = self.files[self.split][index] # if -1, wrap-around
                img = PIL.Image.open(img_file)

        if self.im_size > 0:
        	# Scales image to a square of default size 250x250
        	scaled_dim = (self.im_size.astype(np.int32), 
        				  self.im_size.astype(np.int32))
        	img = img.resize(scaled_dim, PIL.Image.BILINEAR)

        label = 1 # TODO: read in a class label for each image

        img = np.array(img, dtype=np.uint8)
        im_out = torch.from_numpy(im_out).float()
        im_out = im_out.permute(2,0,1) # C x H x W

        return im_out, label



class LFWDataset(data.Dataset):
    '''
        Dataset subclass for loading LFW images in PyTorch.
        This returns multiple images in a batch.
    '''

    def __init__(self, path_list, issame_list, transforms, split = 'test'):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to LFW images
        ''' 
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] =  path_list
        self.pair_label = issame_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if DEBUG:
            print(img_file)
        im_out = self.transforms(img)
        return im_out



class IJBADataset(data.Dataset):
    '''
        Dataset subclass for loading IJB-A images in PyTorch.
        This returns multiple images in a batch.
        Path_list -- full paths to cropped images saved as <sighting_id>.jpg 
    '''
    def __init__(self, path_list, transforms, split=1):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to IJB-A images of one split  
        ''' 
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] =  path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        if DEBUG:
            print(img_file)
        im_out = self.transforms(img)
        return im_out


