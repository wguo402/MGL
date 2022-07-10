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
from loss.triplet_loss import CrossEntropyLabelSmooth
from loss.center_loss import CenterLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

N = 256
IMAGE_PATH = '/home1/food/food-101/images/'
DIR_TRAIN_IMAGES = '/home1/food/food-101/spilted_train.txt'
DIR_TEST_IMAGES = '/home1/food/food-101/spilted_test.txt'

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


class Share_convs(torch.nn.Module):
    def __init__(self, resnet_conv, num_class):
        super(Share_convs, self).__init__()
        self.convout_dimension=2048
        self.resnet_conv = resnet_conv
        self.fc = torch.nn.Linear(self.convout_dimension, num_class)
        self.BN = nn.BatchNorm1d(2048)

    def forward(self, x):
        feature = self.resnet_conv(x)
        feature = self.BN(feature)
        out = self.fc(feature)

        return feature,out

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []

        for line in data_txt:
            line = line.strip()
            words = line.split()

            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        try:

            img = self.loader(os.path.join(IMAGE_PATH, img_name))
            if self.transform is not None:
                img = self.transform(img)
        except:
            img = np.zeros((256, 256, 3), dtype=float)
            img = PIL.Image.fromarray(np.uint8(img))
            if self.transform is not None:
                img = self.transform(img)
            print('erro picture:', img_name)
        return img, label


train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES , transform=test_transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64*4, shuffle=True,  num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=64*4,  shuffle=False, num_workers=4)

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
dataloaders = {'train':train_loader,'test':test_loader}
dataset_sizes = {'train': len(train_dataset),'test': len(test_dataset)}
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
def train_model(model, criterion, optimizer, centerLoss, scheduler,num_epochs=60):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            # phase='test'
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            pbar = tqdm(dataloaders[phase])
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    feature,outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss_center = centerLoss(feature,labels)*0.0005
                    loss = loss_center+loss

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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #torch.save(best_model_wts)
    model.load_state_dict(best_model_wts)
    #model.save_state_dict('inception_v3.pt')
    torch.save(model.state_dict(), 'resnet101_v3.pth')
    return model

# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.


model_ft = models.resnet101(pretrained=None)
pretrainde_dict = (torch.load('/home/guow/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'))
newmodel_dict = model_ft.state_dict()
pretrainde_dict = {k:v for k,v in pretrainde_dict.items() if k in newmodel_dict}
newmodel_dict.update(pretrainde_dict)
model_ft.load_state_dict(newmodel_dict)

#kwargs = {'num_classes':101}
#model_ft = models.alexnet(pretrained=True, **kwargs)

# model_ft.fc = nn.Linear(2048, 101)

model_ft = Share_convs(model_ft,101)
#num_ftrs = model_ft.classifier[6].in_features
#model_ft.classifier[6].out_features = 101

model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('resnet101_v3.pth'))

criterion = CrossEntropyLabelSmooth(101)
# criterion = nn.CrossEntropyLoss()
centerLoss = CenterLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001)
#optimizer_ft = optim.Adam(model_ft.module.fc.parameters())

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Train and evaluate
model_ft = train_model(model_ft, criterion, optimizer_ft, centerLoss,exp_lr_scheduler,
                       num_epochs=60)


#visualize_model(model_ft)


'''
# ConvNet as fixed feature extractor

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()
'''
