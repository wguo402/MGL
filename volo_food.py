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

from utils_ import create_logger
from config import config
import hr_models
#from adabelief_pytorch import AdaBelief
from loss.triplet_loss import CrossEntropyLabelSmooth
from solver import WarmupMultiStepLR
from load_data_set import load_data

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from models.volo import *
from utils import load_pretrained_weights



from apex import amp






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


train_dataset = load_data('/data/food/train/',train_transforms)
# test_dataset =  load_data_set('/data/food/Test/',test_transforms)
val_dataset =  load_data('/data/food/Val/',test_transforms)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32*3, shuffle=True,  num_workers=4)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=48,  shuffle=False, num_workers=2)#48
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,  batch_size=32*3,  shuffle=False, num_workers=4)#48
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
dataloaders = {'train':train_loader,'test':val_loader}
dataset_sizes = {'train': len(train_dataset),'test': len(val_loader)}
# class_names = train_dataset['train'].classes
#
# print(class_names)

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


# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            elif phase=='test' and epoch%2==0:
                model.eval()
            else:
                break  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            pbar = tqdm(dataloaders[phase])
            for inputs, labels,_ in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # pbar.set_description("loss %s" % loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        pbar.set_description("loss %s" % loss.item())
                        # print('{} Loss: {:.4f} '.format(
                        #     phase, loss.item()))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'volo_food.pth')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #torch.save(best_model_wts)
    model.load_state_dict(best_model_wts)
    #model.save_state_dict('inception_v3.pt')
    # torch.save(model.state_dict(), 'resnet101_v3.pth')
    torch.save(model.state_dict(), 'volo_food_food.pth')
    return model




# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.

# model_ft = models.resnet101(pretrained=True)

model_ft = volo_d3().cuda()




load_pretrained_weights(model_ft, '/home/cv1/food/volo/d3_448_86.3.pth.tar', use_ema=False,
                        strict=False, num_classes=1000)


criterion = nn.CrossEntropyLoss().cuda()
# criterion = CrossEntropyLabelSmooth(208)
# Observe that all parameters are being optimized
# optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.00001)
#
# scheduler = WarmupMultiStepLR(optimizer_ft, milestones=[5,10,15], gamma=0.1, warmup_factor=0.01,
#                                           warmup_iters=5, warmup_method='linear')#[10,11,12,13,14,15][10,12,14][5,20,40]

# optimizer_ft = AdaBelief(model_ft.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))


#optimizer_ft = optim.Adam(model_ft.module.fc.parameters())



optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.00001)
scheduler = WarmupMultiStepLR(optimizer_ft, milestones=[5,20,40], gamma=0.1, warmup_factor=0.01,
                                          warmup_iters=5, warmup_method='linear')#[10,11,12,13,14,15][10,12,14][5,20,40]



model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level="O1")



# Decay LR by a factor of 0.1 every 5 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Train and evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,
                       num_epochs=40)


#14  [10,12,14]  acc 67.29








