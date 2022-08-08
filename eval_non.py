
import argparse
import sys,os

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from config import  IMAGENET_Test_PATH
# Purifier
from NRP import *
import random

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Targeted Transferable Perturbations')
parser.add_argument('--test_dir', default='../../../data/IN/val')
parser.add_argument('--save_dir', default='../../../data/IN/val')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for evaluation')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--target_model', type=str, default='vgg19_bn', help='Black-Box(unknown) model: SIN, Augmix etc')
# For purification (https://github.com/Muzammal-Naseer/NRP)
parser.add_argument('--NRP', action='store_true', help='Apply Neural Purification to reduce adversarial effect')
parser.add_argument('--ngpu', type=int, default=1,
                    help='Number of used GPUs (0 = CPU) (default: 1)')
args = parser.parse_args()
print(args)
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()


folder = os.path.exists('./non_subsrc')
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs('./non_subsrc')
logfile = 'non_subsrc/{}_NRP_{}_eps_{}.log'.format(args.target_model, args.NRP,args.eps)


logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

eps = args.eps/255.0
MODE = "bilinear"
# Load pretrained Generator
# netG = GeneratorResnet()

if args.target_model == 'inception_v3':
    scale_size = 300
    img_size = 299
else:
    scale_size = 256
    img_size = 224


# Load Targeted Model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

if args.target_model in model_names:
    model = models.__dict__[args.target_model](pretrained=True)
elif args.target_model == 'SIN':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/resnet50_train_60_epochs-c8e5653e.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
elif args.target_model == 'Augmix':
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/checkpoint.pth.tar')
    model.load_state_dict(checkpoint["state_dict"])
else:
    assert (args.target_model in model_names), 'Please provide correct target model names: {}'.format(model_names)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(2)))
model = model.cuda()
model.eval()


if args.NRP:
    purifier = NRP(3, 3, 64, 23)
    purifier.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
    purifier = purifier.to(device)

####################
# Data
####################
# Input dimensions
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
import numpy as np

test_dir = IMAGENET_Test_PATH

test_set = datasets.ImageFolder(test_dir, data_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
test_size = len(test_set)


noise = np.load('{}/mim_tg_-1_{}_{}.npy'.format(args.save_dir,args.num_models,args.src))
noise = torch.as_tensor(noise).cuda()
logger.info('{}/mim_tg_-1_{}_{}.npy'.format(args.save_dir,args.num_models,args.src))

fool=0
def add_noise(img):
    n = noise
    img = img
    n = F.interpolate(
        n.unsqueeze(0), mode=MODE, size=tuple(img.shape[-2:]), align_corners=True
    ).squeeze()
    return torch.clamp(img + n, 0, 1)
for i, (img, label) in enumerate(test_loader):
    img, label = img.cuda(), label.cuda()
    adv = add_noise(img)
    adv = torch.clamp(adv, 0.0, 1.0)
    if args.NRP:
        # Purify Adversary
        adv = purifier(adv).detach()
    out = model(normalize(adv.clone().detach()))
    img_out = model(normalize(img.clone().detach()))
    fool += torch.sum(out.argmax(dim=-1) != img_out.argmax(dim=-1)).item()

logger.info(' %d     %.4f\t ', int(eps * 255), fool / test_size)