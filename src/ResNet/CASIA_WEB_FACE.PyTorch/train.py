import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import PIL.Image
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import utils
import gc
import sklearn.metrics
from scipy.optimize import brentq
from scipy import interpolate


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



class Trainer(object):

    # -----------------------------------------------------------------------------
    def __init__(self, cuda, model, criterion, optimizer, init_lr,
                 train_loader, val_loader, issame_list, out, max_iter,
                 lr_decay_epoch=None, interval_validate=None):
    # -----------------------------------------------------------------------------
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.issame_list = issame_list
        self.best_roc_auc = 0
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('US/Eastern'))

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'valid/roc_auc',
            'valid/roc_eer',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        


    # -----------------------------------------------------------------------------
    def validate(self):
    # -----------------------------------------------------------------------------
        training = self.model.training
        self.model.eval()
    # -----------------------------------------------------------------------------
    #   feature extraction
    # -----------------------------------------------------------------------------
    #   convert the trained network into a "feature extractor"
        feature_map = list(self.model.children())
        feature_map.pop()
        extractor = nn.Sequential(*feature_map)
        features = []
        for batch_idx, images in tqdm.tqdm(enumerate(self.val_loader),
                                           total=len(self.val_loader),
                                           desc='Extracting features'):
            x = Variable(images, requires_grad=False)  # test-time memory conservation
            if self.cuda:
                x = x.cuda()
            feat = extractor(x)
            if self.cuda:
                feat = feat.data.cpu()
            else:
                feat = feat.data
            features.append(feat)

        features = torch.stack(features)
        sz = features.size()
        features = features.view(sz[0] * sz[1], sz[2])
        features = F.normalize(features, p=2, dim=1)  # L2-normalize
    # TODO - cache features
    # -----------------------------------------------------------------------------
    #   verification
    # -----------------------------------------------------------------------------
        num_feat = features.size()[0]
        feat_pair1 = features[np.arange(0, num_feat, 2), :]
        feat_pair2 = features[np.arange(1, num_feat, 2), :]
        feat_dist = (feat_pair1 - feat_pair2).norm(p=2, dim=1)
        feat_dist = feat_dist.numpy()

    #   eval metrics
        scores = -feat_dist
        gt = np.asarray(self.issame_list)

    #   10 fold
        fold_size = 600  # 600 pairs in each fold
        roc_auc = np.zeros(10)
        roc_eer = np.zeros(10)

        for i in tqdm.tqdm(range(10)):
            start = i * fold_size
            end = (i + 1) * fold_size
            scores_fold = scores[start:end]
            gt_fold = gt[start:end]
            roc_auc[i] = sklearn.metrics.roc_auc_score(gt_fold, scores_fold)
            fpr, tpr, _ = sklearn.metrics.roc_curve(gt_fold, scores_fold)
            # EER calc: https://yangcha.github.io/EER-ROC/
            roc_eer[i] = brentq(
                lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

        print(('LFW VAL AUC: %0.4f +/- %0.4f, LFW VAL EER: %0.4f +/- %0.4f') %
              (np.mean(roc_auc), np.std(roc_auc),
               np.mean(roc_eer), np.std(roc_eer)))
        val_roc_auc = np.mean(roc_auc)
        val_roc_eer = np.mean(roc_eer)

        # Logging
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('US/Eastern')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 2 + \
                  [val_roc_auc, val_roc_eer] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')


        # Saving the best performing model
        is_best = val_roc_auc > self.best_roc_auc
        if is_best:
            self.best_roc_auc = val_roc_auc

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_roc_auc': self.best_roc_auc,
        }, osp.join(self.out, 'checkpoint.pth.tar'))

        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()



    # -----------------------------------------------------------------------------
    def train_epoch(self):
    # -----------------------------------------------------------------------------
        self.model.train()
        n_class = len(self.train_loader.dataset.classes)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            if batch_idx == len(self.train_loader)-1:
                break # discard last batch in epoch (unequal batch-sizes mess up BatchNorm)

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            # Computing Losses
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            score = self.model(data)  # batch_size x num_class

            loss = self.criterion(score, target)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is NaN while training')
            # print list(self.model.parameters())[0].grad

            # Gradient descent
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Computing metrics
            lbl_pred = score.data.max(1)[1].cpu().numpy()
            lbl_pred = lbl_pred.squeeze()
            lbl_true = target.data.cpu()
            lbl_true = np.squeeze(lbl_true.numpy())
            train_accu = self.eval_metric([lbl_pred], [lbl_true])

            # Logging
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('US/Eastern')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                      [train_accu] + [''] * 2 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
                print('\nEpoch: ' + str(self.epoch) + ' Iter: ' + str(self.iteration) + \
                         ' Loss: ' + str(loss.data[0]))

            if self.iteration >= self.max_iter:
                break


    # -----------------------------------------------------------------------------
    def eval_metric(self, lbl_pred, lbl_true):
    # -----------------------------------------------------------------------------
        # Over-all accuracy
        # TODO: per-class accuracy
        accu = []
        for lt, lp in zip(lbl_true, lbl_pred):
            accu.append(np.mean(lt == lp))
        return np.mean(accu)


    # -----------------------------------------------------------------------------
    def train(self):
    # -----------------------------------------------------------------------------
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        print('Number of iters in an epoch: %d' % len(self.train_loader))
        print('Total epochs: %d' % max_epoch)

        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train epochs', ncols=80, leave=True):
            self.epoch = epoch

            if self.lr_decay_epoch is None:
                pass
            else:
                assert self.lr_decay_epoch < max_epoch
                lr_scheduler(self.optim, self.epoch, 
                             init_lr=self.init_lr, 
                             lr_decay_epoch=self.lr_decay_epoch)

            self.train_epoch()
            if self.iteration >= self.max_iter:
                break