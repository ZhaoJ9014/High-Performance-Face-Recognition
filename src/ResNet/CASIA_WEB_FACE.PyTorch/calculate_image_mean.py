import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import yaml
import tqdm

here = os.path.dirname(os.path.abspath(__file__))  # output folder is located here
root_dir, _ = os.path.split(here)
import sys

sys.path.append(root_dir)

torch.manual_seed(1337)

'''
Calculate mean R, G, B values for the dataset
--------------------------------------------------
'''

dataset_path = '/home/zhaojian/zhaojian/DATA/CASIA_WEB_FACE_Resized'

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1337)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # enable if all images are same size

# -----------------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------------
data_root = dataset_path
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),  # smaller side resized
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Data loaders
dataset = datasets.ImageFolder(data_root, train_transform)
train_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=128, **kwargs)

rgb_mean = []
for batch_idx, (images, lbl) in tqdm.tqdm(enumerate(train_loader),
                                              total=len(train_loader),
                                              desc='Sampling images'):
    rgb_mean.append(((images.mean(dim=0)).mean(dim=-1)).mean(dim=-1))
    if batch_idx == 100:
        break

print(len(rgb_mean))
rgb_mean = torch.mean(torch.stack(rgb_mean, dim=1), dim=1)
print(rgb_mean)

res = {}
res['R'] = rgb_mean[0]
res['G'] = rgb_mean[1]
res['B'] = rgb_mean[2]

with open(os.path.join(here, 'mean_rgb.yaml'), 'w') as f:
    yaml.dump(res, f, default_flow_style=False)
