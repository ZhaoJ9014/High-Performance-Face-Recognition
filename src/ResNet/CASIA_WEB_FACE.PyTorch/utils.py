from __future__ import division

import math
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import scipy.ndimage
import six
import skimage
import skimage.color
from skimage import img_as_ubyte
import os
import os.path as osp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import scipy.signal

def make_weights_for_balanced_classes(images, nclasses):  
    '''
        Make a vector of weights for each image in the dataset, based 
        on class frequency. The returned vector of weights can be used 
        to create a WeightedRandomSampler for a DataLoader to have 
        class balancing when sampling for a training batch. 
            images - torchvisionDataset.imgs 
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3                      
    '''
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))  # total number of images                  
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def get_vgg_class_counts(log_path):
    '''  Dict of class frequencies from pre-computed text file  '''
    data_1 = np.genfromtxt(log_path, dtype=None)
    class_names = [x[0] for x in data_1]
    class_counts = [x[1] for x in data_1]
    class_count_dict = dict(zip(class_names, class_counts))
    return class_count_dict

def plot_log_csv(log_path):
    log_dir, _ = osp.split(log_path)
    dat = np.genfromtxt(log_path, names=True, 
                        delimiter=',', autostrip=True)

    train_loss =  dat['trainloss']
    train_loss_sel = ~np.isnan(train_loss)
    train_loss = train_loss[train_loss_sel]
    iter_train_loss = dat['iteration'][train_loss_sel]

    train_acc = dat['trainacc']
    train_acc_sel = ~np.isnan(train_acc)
    train_acc = train_acc[train_acc_sel]
    iter_train_acc = dat['iteration'][train_acc_sel]

    val_loss =  dat['validloss']
    val_loss_sel = ~np.isnan(val_loss)
    val_loss = val_loss[val_loss_sel]
    iter_val_loss = dat['iteration'][val_loss_sel]

    mean_iu = dat['validacc']
    mean_iu_sel = ~np.isnan(mean_iu)
    mean_iu = mean_iu[mean_iu_sel]
    iter_mean_iu = dat['iteration'][mean_iu_sel]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.plot(iter_train_acc, train_acc, label='train')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(iter_mean_iu, mean_iu, label='val')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(iter_train_loss, train_loss, label='train')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(iter_val_loss, val_loss, label='val')
    plt.xlabel('iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(osp.join(log_dir, 'log_plots.png'), bbox_inches='tight')

def plot_log(log_path):
    log_dir, _ = osp.split(log_path)
    epoch = []
    iteration = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    g = lambda x: x if x!='' else float('nan')
    reader = csv.reader( open(log_path, 'rb'))
    next(reader)  # Skip header row.
    for line in reader:
        line_fields = [g(x) for x in line]
        epoch.append(float(line_fields[0]))
        iteration.append(float(line_fields[1]))
        train_loss.append(float(line_fields[2]))
        train_acc.append(float(line_fields[3]))
        val_loss.append(float(line_fields[4]))
        val_acc.append(float(line_fields[5]))

    epoch = np.array(epoch)
    iteration = np.array(iteration)
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)

    train_loss_sel = ~np.isnan(train_loss)
    train_loss = train_loss[train_loss_sel]
    iter_train_loss = iteration[train_loss_sel]

    train_acc_sel = ~np.isnan(train_acc)
    train_acc = train_acc[train_acc_sel]
    iter_train_acc = iteration[train_acc_sel]

    val_loss_sel = ~np.isnan(val_loss)
    val_loss = val_loss[val_loss_sel]
    iter_val_loss = iteration[val_loss_sel]

    val_acc_sel = ~np.isnan(val_acc)
    val_acc = val_acc[val_acc_sel]
    iter_val_acc = iteration[val_acc_sel]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.plot(iter_train_acc, train_acc, label='train', alpha=0.5, color='C0')
    box_pts = np.rint(np.sqrt(len(train_acc))).astype(np.int)
    plt.plot(iter_train_acc, savgol_smooth(train_acc, box_pts), color='C0')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.title('Training')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(iter_val_acc, val_acc, label='val', alpha=0.5, color='C1')
    box_pts = np.rint(np.sqrt(len(val_acc))).astype(np.int)
    plt.plot(iter_val_acc, savgol_smooth(val_acc, box_pts), color='C1')
    plt.grid()
    plt.legend()
    plt.title('Validation')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(iter_train_loss, train_loss, label='train', alpha=0.5, color='C0')
    box_pts = np.rint(np.sqrt(len(train_loss))).astype(np.int)
    plt.plot(iter_train_loss, savgol_smooth(train_loss, box_pts), color='C0')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(iter_val_loss, val_loss, label='val', alpha=0.5, color='C1')
    box_pts = np.rint(np.sqrt(len(val_loss))).astype(np.int)
    plt.plot(iter_val_loss, savgol_smooth(val_loss, box_pts), color='C1')
    plt.xlabel('iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(osp.join(log_dir, 'log_plots.png'), bbox_inches='tight')

def savgol_smooth(y, box_pts):
    # use the Savitzky-Golay filter for 1-D smoothing
    if box_pts % 2 == 0:
        box_pts += 1
    y_smooth = scipy.signal.savgol_filter(y, box_pts, 2)
    return y_smooth

# -----------------------------------------------------------------------------
#   LFW helper code from FaceNet: https://github.com/davidsandberg/facenet
# ----------------------------------------------------------------------------- 

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename, lfw_flag=True):
    pairs = []
    with open(pairs_filename, 'r') as f:
        if lfw_flag:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        else:
            for line in f.readlines():
                pair = line.strip().split()
                pairs.append(pair)      
    return np.array(pairs)