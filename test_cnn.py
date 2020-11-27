# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 07:41:40 2020

@author: timot
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as tmodels
from IPython.core.debugger import set_trace
from pdb import set_trace

import os
import numpy as np
import time
import utils
from matplotlib import pyplot as plt

from train import evaluate

#%% Inputs

testdir = ['/content/drive/My Drive/ECE6254_Project/Code/Patterns/Test']
resume = '/content/drive/My Drive/ECE6254_Project/checkpoints/AlexNet/model_best.pth.tar'
batch_size = 256
workers = 4
resize_val = 2**8

#%% Import model

model = tmodels.alexnet(pretrained=True)
model.classifier[4] = nn.Linear(4096,1024)
model.classifier[6] = nn.Linear(1024,26)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
print("=> creating model %s " % model.__class__.__name__)

# define loss function (criterion) and optimizer
if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume)
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {}, best_prec1 @ Source {})"
      .format(resume, checkpoint['epoch'], best_prec1))

#%% Iterate and test

iterate_dirs = ['-4','-2','00','02','04','06','08','10','12','14','16','18','20']

# Evaluate
accuracy_all = []
for idx,i_dB in enumerate(iterate_dirs):
    sub_dir = i_dB+'dB_50Its_M5_Size8_Phase_Test'
    i_testdir = [os.path.join(testdir[0],sub_dir)]
    # Transform for AlexNet
    transform = transforms.Compose([
        transforms.Resize(resize_val), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    
    test_dataset = utils.ArrayDataset(i_testdir, transform)
    
    # Load dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=True)
    
    # Evaluate
    loss, top1, top5 = evaluate(test_loader, model, criterion)
    if isinstance(top1, int):
        accuracy_all.append(top1)
    else:
        accuracy_all.append(top1.item())
    print('{sub_dir_str}: {top1:.3f}'.format(sub_dir_str=sub_dir,top1=top1))

accuracy_all = np.array(accuracy_all)
np.savetxt('/content/drive/My Drive/ECE6254_Project/accuracy_all.csv',accuracy_all,delimiter=',')