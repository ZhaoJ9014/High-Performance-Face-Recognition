import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import utils
import data_loader
import time
import copy
import tqdm
import sklearn.metrics
from scipy.optimize import brentq
from scipy import interpolate
from config import configurations
from center_loss import CenterLoss

#======= Parameters and DataLoaders =======#
cfg = configurations[1]

SEED = cfg['SEED']
torch.manual_seed(SEED)

LR_SOFTMAX = cfg['LR_SOFTMAX'] # LR for softmax loss
LR_CENTER = cfg['LR_CENTER'] # LR for center loss
ALPHA = cfg['ALPHA'] # Weight for center loss
TRAIN_BATCH_SIZE = cfg['TRAIN_BATCH_SIZE']
VAL_BATCH_SIZE = cfg['VAL_BATCH_SIZE']
NUM_EPOCHS = cfg['NUM_EPOCHS']
WEIGHT_DECAY = cfg['WEIGHT_DECAY']

RGB_MEAN = cfg['RGB_MEAN']
RGB_STD = cfg['RGB_STD']

MODEL_NAME = cfg['MODEL_NAME']
TRAIN_PATH = cfg['TRAIN_PATH']
VAL_PATH = cfg['VAL_PATH']
PAIR_TEXT_PATH = cfg['PAIR_TEXT_PATH']
FILE_EXT = cfg['FILE_EXT']

train_transform = transforms.Compose([
    transforms.Resize(256),  # smaller side resized
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN,
                         std=RGB_STD),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN,
                         std=RGB_STD),
])

dataset_train = datasets.ImageFolder(TRAIN_PATH, train_transform)

# For unbalanced dataset we create a weighted sampler
#   * Balanced class sampling: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
weights = utils.make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, sampler=sampler, drop_last=True)
num_class = len(train_loader.dataset.classes)
print('Number of Training Classes: %d' % num_class)

pairs = utils.read_pairs(PAIR_TEXT_PATH)
path_list, issame_list = utils.get_paths(VAL_PATH, pairs, FILE_EXT)
val_loader = torch.utils.data.DataLoader(data_loader.LFWDataset(path_list, issame_list, val_transform), batch_size=VAL_BATCH_SIZE, shuffle=False)

#======= Model & Loss & Optimizer =======#
if MODEL_NAME.lower()=='resnet18':
    model = torchvision.models.resnet18(pretrained=True)
elif MODEL_NAME.lower()=='resnet34':
    model = torchvision.models.resnet34(pretrained=True)
elif MODEL_NAME.lower()=='resnet50':
    model = torchvision.models.resnet50(pretrained=True)
elif MODEL_NAME.lower()=='resnet101':
    model = torchvision.models.resnet101(pretrained=True)
elif MODEL_NAME.lower()=='resnet152':
    model = torchvision.models.resnet152(pretrained=True)
else:
    raise NotImplementedError('Model: ResNet-18, 34, 50, 101, 152')

if type(model.fc) == nn.modules.linear.Linear:
    # Check if final fc layer sizes match num_class
    if not model.fc.weight.size()[0] == num_class:

        # Replace last layer
        print('The Original FC Layer: ', model.fc)
        model.fc = nn.Linear(2048, num_class)
        print('is Replaced with A New One: ', model.fc)
    else:
        pass
else:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# Softmax loss
criterion_softmax = nn.CrossEntropyLoss()
criterion_softmax = criterion_softmax.to(device)

# Center loss
class CenterFeature(nn.Module):
    def __init__(self, submodule):
        super(CenterFeature, self).__init__()
        self.submodule = submodule
        self.extracted_layer = ["avgpool"]

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layer:
                outputs.append(x)
        return outputs

criterion_centerloss = CenterLoss(num_classes=num_class, feat_dim=2048, use_gpu=True)

# Optimizer
if cfg['OPTIM'].lower()=='adam':
    optimizer_softmax = optim.Adam(model.parameters(), lr=LR_SOFTMAX, weight_decay=WEIGHT_DECAY)
    optimizer_center = optim.Adam(criterion_centerloss.parameters(), lr=LR_CENTER)
else:
    raise NotImplementedError('Optimizer: Adam')

# Decay LR by a factor of 0.1 every 10 epochs
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#======= Train & Validation & Saving Checkpoint =======#
train_batch_loss_history = []
train_batch_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_roc_auc = 0.0
since = time.time()

for epoch in range(NUM_EPOCHS):

        since_epoch = time.time()
        running_loss = 0.0
        running_corrects = 0

        # Set model to training mode
        model.train()

        # scheduler.step()

        # Iterate over data.
        for step, (inputs, labels) in enumerate(train_loader):
            print('Epoch {}/{} Batch {}/{}'.format(epoch, NUM_EPOCHS - 1, step, len(train_loader) - 1))
            since_step = time.time()

            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            # Forward
            # Track history if only in train
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss_softmax = criterion_softmax(outputs, labels)
            _, preds = torch.max(outputs, 1)

            model_center = model.module
            myexactor = CenterFeature(model_center)
            features_center = myexactor(inputs)[0].squeeze()
            features_center = F.normalize(features_center, p=2, dim=1)  # L2-normalize
            loss_center = criterion_centerloss(features_center, labels)

            loss = loss_softmax + loss_center * ALPHA

            # Zero the parameter gradients
            optimizer_softmax.zero_grad()
            optimizer_center.zero_grad()

            # backward + optimize only if in training phase
            loss.backward()

            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
            for param in criterion_centerloss.parameters():
                param.grad.data *= (1. / ALPHA)

            optimizer_softmax.step()
            optimizer_center.step()

            time_elapsed = time.time() - since_step
            print('Train Batch Loss: {:.4f} Train Batch Softmax Loss: {:.4f} Train Batch Center Loss: {:.4f} Acc: {:.4f} Elapsed: {:.0f}m {:.0f}s'.format(loss.data.item(), loss_softmax.data.item(), loss_center.data.item() * 0.01, torch.sum(preds == labels.data).double() / inputs.data.size(0), time_elapsed // 60, time_elapsed % 60))
            train_batch_loss_history.append(loss.data.item())
            train_batch_acc_history.append(torch.sum(preds == labels.data).double() / inputs.data.size(0))

            # Statistics
            running_loss += loss.data.item() * inputs.data.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        time_elapsed = time.time() - since_epoch
        print('Train Epoch Loss: {:.4f} Acc: {:.4f} Elapsed: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))

        # Set model to evaluate mode
        model_eval = model.module  # get network module from inside its DataParallel wrapper
        model_eval = model_eval.to(device)
        model_eval.eval()

        # Feature extraction
        # Convert the trained network into a "feature extractor"
        extractor = nn.Sequential(*list(model_eval.children())[:-1])
        features = []
        for batch_idx, images in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='Extracting features'):
            x = Variable(images).to(device) # Test-time memory conservation
            feat = extractor(x).data.cpu()
            features.append(feat)
        features = torch.stack(features)
        sz = features.size()

        features = features.view(sz[0] * sz[1], sz[2])
        features = F.normalize(features, p=2, dim=1)  # L2-normalize
        # Verification
        num_feat = features.size()[0]
        feat_pair1 = features[np.arange(0, num_feat, 2), :]
        feat_pair2 = features[np.arange(1, num_feat, 2), :]
        feat_dist = (feat_pair1 - feat_pair2).norm(p=2, dim=1)
        feat_dist = feat_dist.numpy()

        # Eval metrics
        scores = -feat_dist
        gt = np.asarray(issame_list)

        # 10 fold
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

        print(('LFW VAL AUC: %0.4f +/- %0.4f, LFW VAL EER: %0.4f +/- %0.4f') % (np.mean(roc_auc), np.std(roc_auc), np.mean(roc_eer), np.std(roc_eer)))
        epoch_val_roc_auc = np.mean(roc_auc)
        epoch_val_roc_eer = np.mean(roc_eer)

        if epoch_val_roc_auc > best_roc_auc:
            best_roc_auc = epoch_val_roc_auc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'arch': model.__class__.__name__,
                'optim_softmax_state_dict': optimizer_softmax.state_dict(),
                'optim_center_state_dict': optimizer_center.state_dict(),
                'model_state_dict': model.state_dict(),
                'train_epoch_loss': epoch_loss,
                'train_epoch_acc': epoch_acc,
                'best_roc_auc': best_roc_auc,
            }, './models/{}_CASIA-WEB-FACE-Aligned_Epoch_{}_LfwAUC_{}.tar'.format(MODEL_NAME, epoch, best_roc_auc))
        print('Current Best val ROC AUC: {:4f}'.format(best_roc_auc))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save train_batch_loss_history and train_batch_acc_history
with open('./logs/{}_train_batch_loss_history_Aligned.txt'.format(MODEL_NAME), 'w') as f:
    for item in train_batch_loss_history:
        f.write("%s\n" % item)
with open('./logs/{}_train_batch_acc_history_Aligned.txt'.format(MODEL_NAME), 'w') as f:
    for item in train_batch_acc_history:
        f.write("%s\n" % item)
