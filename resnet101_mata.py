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
from trainer import train,validate
from trainer import Share_convs
from opts import opts
import shutil
import learn2learn as l2l
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

N = 256
LR = 0.001
Momentum=0.9
Weight_decay=0.0001

IMAGE_PATH = '/home1/food/food-101/images/'
Auxiliary_path = '/home1/food/isiaFood_200'
DIR_TRAIN_IMAGES = '/home1/food/food-101/spilted_train.txt'
DIR_TEST_IMAGES = '/home1/food/food-101/spilted_test.txt'
Auxiliary_DIR_TRAIN_IMAGES = '/home1/food/isiaFood_200/train_finetune_v2.txt'
Auxiliary_DIR_VAL_IMAGES = '/home1/food/isiaFood_200/val_finetune_v2.txt'
Auxiliary_DIR_TEST_IMAGES = '/home1/food/isiaFood_200/test_finetune_v2.txt'
best_prec1 = 0
args = opts()

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

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, image_paht, txt_dir, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        self.image_path=image_paht

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

            img = self.loader(os.path.join(self.image_path, img_name))
            if self.transform is not None:
                img = self.transform(img)
        except:
            img = np.zeros((256, 256, 3), dtype=float)
            img = PIL.Image.fromarray(np.uint8(img))
            if self.transform is not None:
                img = self.transform(img)
            print('erro picture:', img_name)
        return img, label


train_dataset = MyDataset(Auxiliary_path, txt_dir=Auxiliary_DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(Auxiliary_path, txt_dir=Auxiliary_DIR_TEST_IMAGES , transform=test_transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=48*2, shuffle=True,  num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=48*2,  shuffle=False, num_workers=2)

auxiliary_train_dataset = MyDataset(Auxiliary_path,txt_dir=Auxiliary_DIR_VAL_IMAGES , transform=train_transforms)
# auxiliary_test_dataset = MyDataset(Auxiliary_path,txt_dir=Auxiliary_DIR_TEST_IMAGES , transform=test_transforms)
auxiliary_train_loader = torch.utils.data.DataLoader(dataset=auxiliary_train_dataset, batch_size=48, shuffle=True,  num_workers=4)
# auxiliary_test_loader = torch.utils.data.DataLoader(dataset=auxiliary_test_dataset,  batch_size=48,  shuffle=False, num_workers=2)



# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
dataloaders = {'train':train_loader,'test':test_loader}
dataset_sizes = {'train': len(train_loader),'test': len(test_loader)}

auxiliary_dataloaders = {'train':train_loader,'test':test_loader}
auxiliary_dataset_sizes = {'train': len(train_loader),'test': len(test_loader)}
# class_names = train_dataset['train'].classes
#
# print(class_names)
train_loader_source = auxiliary_train_loader
train_loader_source_batch = enumerate(auxiliary_train_loader)
train_loader_target = train_loader
train_loader_target_batch = enumerate(train_loader)

val_loader_source = test_loader
val_loader_target = test_loader


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
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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


# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.

def save_checkpoint(state, is_best, args, epoch):
    filename = str(epoch) + 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))

model_ft = models.resnet101(pretrained=None)
pretrainde_dict = (torch.load('/home/guow/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'))
newmodel_dict = model_ft.state_dict()
pretrainde_dict = {k:v for k,v in pretrainde_dict.items() if k in newmodel_dict}
newmodel_dict.update(pretrainde_dict)
model_ft.load_state_dict(newmodel_dict)

model_source = Share_convs(model_ft,200)
# model_target = model_source

model_source = torch.nn.DataParallel(model_source).cuda()
model_target = model_source



# model_ft = nn.DataParallel(model_ft)
# model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.0001)

optimizer = torch.optim.SGD([
                {'params': model_source.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_source.module.fc.parameters(), 'name': 'pre-trained'},
                # {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
            ],lr=LR,momentum=Momentum,weight_decay=Weight_decay)
# Decay LR by a factor of 0.1 every 5 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Train and evaluate
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25)
resume = '/home/guow/food/Finetune-pytorch/Food101-resnet101/checkpoints_resnet34_stanford_dogs_128Timg_imagenet_128Simg_Meta_train_Lr0.001_1/model_best.pth.tar'
if os.path.isfile(resume):
    # raise ValueError('the resume function is not finished')
    print("==> loading checkpoints '{}'".format(resume))
    checkpoint = torch.load(resume)
    args.start_epoch = checkpoint['epoch']
    if args.meta_sgd:
        meta_train_lr = checkpoint['meta_train_lr']
    best_prec1 = checkpoint['best_prec1']
    model_source.load_state_dict(checkpoint['source_state_dict'])
    model_target.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("==> loaded checkpoint '{}'(epoch {})"
          .format(resume, checkpoint['epoch']))
else:
    raise ValueError('The file to be resumed from is not exited', resume)

for epoch in range(args.start_epoch, args.epochs):
    train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch,
                                                             train_loader_target, train_loader_target_batch,
                                                             model_source, model_target, criterion, optimizer, epoch,
                                                             args, None)

    if (epoch + 1) % 2253 == 0 or (epoch + 1) % args.epochs == 0:

        prec1 = validate(val_loader_source, val_loader_target, model_source, model_target, criterion, epoch, args)

        # record the best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write('     \nTarget_T1 acc: %3f' % (best_prec1))
            log.close()


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'source_state_dict': model_source.state_dict(),
            'target_state_dict': model_target.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args, epoch + 1)

# evaluate on the val data



#visualize_model(model_ft)


