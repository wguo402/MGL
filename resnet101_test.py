# -*- coding: utf-8 -*-

###########################################
# Created by andrew
# 2018/6/12
# borrow some code from pytorch tutorial
# predict label using trained model
###########################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from TestDataset import TestDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/andrew/datasets/food-101/Food101_splited/test/'
class_names = os.listdir(data_dir)
class_names.sort()
print(class_names)

dataset = TestDataset(data_dir + 'chicken_wings/', data_transforms['test'])
assert dataset
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32,
        drop_last=False, shuffle=False, num_workers=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
model_ft = models.resnet101(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 101)

model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('resnet101_v3.pth'))

#####################################################################
# It is very important to turn the model into evaluation mode!!!!
# or the BN and drop out will not fixed and get a very lower accuracy!
######################################################################
model_ft.eval()
show_predict_img = True

for i, imgs in enumerate(dataloader):
    #print(i)
    img_inputs = imgs[0]
    img_inputs.to(device)
    img_names = imgs[1]

    outputs = model_ft(img_inputs)
    _, preds = torch.max(outputs, 1)
    #print(preds)
    pred_class = {}
    for j in range(len(preds)):
        #pred_class.append(class_names[item])
        pred_class[img_names[j]] = class_names[preds[j]]
    print(pred_class)

    if(show_predict_img):
        images_so_far = 0
        num_images = 6
        fig = plt.figure()
        for j in range(img_inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(img_inputs.cpu().data[j])
                if(images_so_far == num_images):
                    time.sleep(200)