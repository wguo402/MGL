# -*- coding: utf-8 -*-

###########################################
# borrow code from pytorch tutorial
# fintune food101 using pretrained resnet101
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
import PIL
from PIL import ImageFile
from tqdm import tqdm

from utils import create_logger
from config import config
import hr_models
#from adabelief_pytorch import AdaBelief
from loss.triplet_loss import CrossEntropyLabelSmooth
from solver import WarmupMultiStepLR
from load_data_set import load_data
import pandas as pd

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

N = 256


experimnet_name = 'cls_hrnet_w64.yaml'

config.defrost()
config.merge_from_file(experimnet_name)

train_transforms = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                     transforms.Resize((N, N)),
                     transforms.RandomCrop((224,224)),
                     transforms.ToTensor(),
                     normalize
                  ])

test_transforms = transforms.Compose([
                    transforms.Resize((N, N)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    normalize
                  ])

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')



# test_dataset =  load_data_set('/data/food/Test/',test_transforms)
val_dataset =  load_data('/data/food/test/',test_transforms)



# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=48,  shuffle=False, num_workers=2)#48
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,  batch_size=48,  shuffle=False, num_workers=4)#48
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
dataloaders = {'test':val_loader}
dataset_sizes = {'test': len(val_loader)}
# class_names = train_dataset['train'].classes
#
# print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# Training the model
def test(model):
    model.eval()

    all_preds=[]
    all_name=[]

    pbar = tqdm(dataloaders['test'])
    count = 1
    for inputs, labels,path_name in pbar:
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_name.append(list(path_name))
        if count>3000:
            break
        count+=1

        # statistics
    pred_array = np.array(all_preds).reshape(-1,1)
    name_array = np.array(all_name).reshape(-1,1)

    result = np.hstack((name_array,pred_array))
    import pandas
    data1 = pandas.DataFrame(result)
    data1.to_csv('sample_submission.csv',index=0,header=['id','predicted'],float_format='%s')
    #np.savetxt("result.csv", result, delimiter=',',fmt='%s')
# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.

# model_ft = models.resnet101(pretrained=True)8
model_ft = eval('hr_models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

model_ft = nn.DataParallel(model_ft)
model_ft.load_state_dict(torch.load('hr_food_food.pth'))


model_ft = test(model_ft)


#14  [10,12,14]  acc 67.29
#22  [10,15,20]  acc 75 0.00001










