import argparse
import datetime
import os
import os.path as osp
import pytz

import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = osp.dirname(osp.abspath(__file__)) # output folder is located here
root_dir,_ = osp.split(here)
import sys
sys.path.append(root_dir)

import train
import models
import utils
from config import configurations
import data_loader




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='resnet50')
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('-d', '--dataset_path', 
                        default='/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE_Aligned')
    parser.add_argument('-m', '--model_path', default=None, 
                        help='Initialize from pre-trained model')
    parser.add_argument('--resume', help='Checkpoint path')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                        help='Add a 512-dim bottleneck layer with L2 normalization')
    args = parser.parse_args()

    # gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir(args.exp_name, args.config, cfg, verbose=False)
    resume = args.resume

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size    



    # -----------------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------------
    #  Images should be arranged like this:
    #   data_root/
    #       class_1/....jpg..
    #       class_2/....jpg.. 
    #       ......./....jpg.. 
    data_root = args.dataset_path
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    RGB_MEAN = [ 0.485, 0.456, 0.406 ]
    RGB_STD = [ 0.229, 0.224, 0.225 ]
    
    # Data transforms
    # http://pytorch.org/docs/master/torchvision/transforms.html
    train_transform = transforms.Compose([
        transforms.Resize(256),  # smaller side resized
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    # Data loaders - using PyTorch built-in objects
    #   loader = DataLoaderClass(DatasetClass)
    #   * `DataLoaderClass` is PyTorch provided torch.utils.data.DataLoader
    #   * `DatasetClass` loads samples from a dataset; can be a standard class 
    #     provided by PyTorch (datasets.ImageFolder) or a custom-made class.
    #      - More info: http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    traindir = osp.join(data_root)
    dataset_train = datasets.ImageFolder(traindir, train_transform)
    
    # For unbalanced dataset we create a weighted sampler
    #   *  Balanced class sampling: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3                     
    weights = utils.make_weights_for_balanced_classes(
                dataset_train.imgs, len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
                    dataset_train, batch_size=cfg['batch_size'], 
                    sampler = sampler, **kwargs)

    file_ext = 'jpg'  # observe, no '.' before jpg
    valdir = '/home/zhaojian/zhaojian/DATA/lfw_Aligned'
    pairs_path = '/home/zhaojian/zhaojian/DATA/pairs.txt'
    pairs = utils.read_pairs(pairs_path)
    path_list, issame_list = utils.get_paths(valdir, pairs, file_ext)
    val_loader = torch.utils.data.DataLoader(
        data_loader.LFWDataset(
            path_list, issame_list, val_transform),
        batch_size=cfg['batch_size'], shuffle=False, **kwargs)

    # print 'dataset classes:' + str(train_loader.dataset.classes)
    num_class = len(train_loader.dataset.classes)
    print('Number of classes: %d' % num_class)



    # -----------------------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------------------
    model = torchvision.models.resnet50(pretrained=True)

    if type(model.fc) == torch.nn.modules.linear.Linear:
        # Check if final fc layer sizes match num_class
        if not model.fc.weight.size()[0] == num_class:
            # Replace last layer
            print(model.fc)
            model.fc = torch.nn.Linear(2048, num_class)
            print(model.fc)
        else:
            pass
    else:
        pass    


    if args.model_path:
        # If existing model is to be loaded from a file
        checkpoint = torch.load(args.model_path) 

        if checkpoint['arch'] == 'DataParallel':
            # if we trained and saved our model using DataParallel
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.module # get network module from inside its DataParallel wrapper
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally add a "bottleneck + L2-norm" layer after GAP-layer
    # TODO -- loading a bottleneck model might be a problem .... do some unit-tests
    if args.bottleneck:
        layers = []
        layers.append(torch.nn.Linear(2048, 512))
        layers.append(nn.BatchNorm2d(512))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(models.NormFeat()) # L2-normalization layer
        layers.append(torch.nn.Linear(512, num_class))
        model.fc = torch.nn.Sequential(*layers)

    # TODO - config options for DataParallel and device_ids
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    if cuda:
        model.cuda()  

    start_epoch = 0
    start_iteration = 0

    # Loss - cross entropy between predicted scores (unnormalized) and class labels (integers)
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    if resume:
        # Resume training from last saved checkpoint
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        pass


    # -----------------------------------------------------------------------------
    # 3. Optimizer
    # -----------------------------------------------------------------------------
    params = filter(lambda p: p.requires_grad, model.parameters()) 
    # Parameters with p.requires_grad=False are not updated during training.
    # This can be specified when defining the nn.Modules during model creation

    if 'optim' in cfg.keys():
        if cfg['optim'].lower()=='sgd':
            optim = torch.optim.SGD(params,
                        lr=cfg['lr'],
                        momentum=cfg['momentum'],
                        weight_decay=cfg['weight_decay'])

        elif cfg['optim'].lower()=='adam':
            optim = torch.optim.Adam(params,
                        lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        else:
            raise NotImplementedError('Optimizers: SGD or Adam')
    else:
        optim = torch.optim.SGD(params,
                    lr=cfg['lr'],
                    momentum=cfg['momentum'],
                    weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])


    # -----------------------------------------------------------------------------
    # 4. Training
    # -----------------------------------------------------------------------------
    trainer = train.Trainer(
        cuda=cuda,
        model=model,
        criterion=criterion,
        optimizer=optim,
        init_lr=cfg['lr'],
        lr_decay_epoch = None,
        train_loader=train_loader,
        val_loader=val_loader,
        issame_list=issame_list,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )

    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()



def get_log_dir(model_name, config_id, cfg, verbose=True):
    # Creates an output directory for each experiment, timestamped
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    if verbose:
        for k, v in cfg.items():
            v = str(v)
            if '/' in v:
                continue
            name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir




if __name__ == '__main__':
    main()