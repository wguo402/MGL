#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load mini-ImageNet, and
    * sample tasks and split them in adaptation and evaluation sets.
To contrast the use of the benchmark interface with directly instantiating mini-ImageNet datasets and tasks, compare with `protonet_miniimagenet.py`.
"""
import os
import random
import numpy as np
import pickle

import torch
from torch import nn, optim

import learn2learn as l2l

from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)

from torchvision import datasets, models, transforms
import PIL
from tqdm import tqdm
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

IMAGE_PATH = '/home1/food/isiaFood_200'
DIR_TRAIN_IMAGES = '/home1/food/isiaFood_200/train_finetune_v2.txt'
DIR_TEST_IMAGES = '/home1/food/isiaFood_200/test_finetune_v2.txt'
DIR_VAL_IMAGES =  '/home1/food/isiaFood_200/val_finetune_v2.txt'
# DIR_VAL_IMAGES =  '/home1/food/isiaFood_200/debug.txt'
print(IMAGE_PATH)
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
N = 256
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
        # self.BN = nn.BatchNorm1d(2048)

    def forward(self, x):
        feature = self.resnet_conv(x)
        # feature = self.BN(feature)
        out = self.fc(feature)

        return out

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

        # img_name, label = self.imgs[index]
        # # print label
        # img = self.loader(os.path.join(IMAGE_PATH, img_name))
        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, label


train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES , transform=test_transforms)
val_dataset = MyDataset(txt_dir=DIR_VAL_IMAGES , transform=train_transforms)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64*4, shuffle=True,  num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=96,  shuffle=False, num_workers=4)




def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        adaptation_error /= len(adaptation_data)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_error /= len(evaluation_data)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.003,
        fast_lr=0.1,
        meta_batch_size=32,
        adaptation_steps=2,
        num_iterations=6000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')




    # Create Tasksets using the benchmark interface
    # tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
    #                                               train_samples=2*shots,
    #                                               train_ways=ways,
    #                                               test_samples=2*shots,
    #                                               test_ways=ways,
    #                                               root='~/data',
    # )
    #

    # train_data_load = l2l.data.MetaDataset(train_dataset)# any PyTorch dataset
    # output_hal = open("train_data_load.pkl", 'wb')
    # str = pickle.dumps(train_data_load)
    # output_hal.write(str)
    # output_hal.close()

    with open("train_data_load.pkl", 'rb') as file:
        train_data_load = pickle.loads(file.read())
    transforms = [  # Easy to define your own transform
        l2l.data.transforms.NWays(train_data_load, n=5),
        l2l.data.transforms.KShots(train_data_load, k=10),
        l2l.data.transforms.LoadData(train_data_load),
    ]
    train_data_load = l2l.data.TaskDataset(train_data_load, transforms, num_tasks=20000)

    # val_dataset_load = l2l.data.MetaDataset(val_dataset)  # any PyTorch dataset

    # output_hal = open("val_data_load.pkl", 'wb')
    # str = pickle.dumps(val_dataset_load)
    # output_hal.write(str)
    # output_hal.close()
    with open("val_data_load.pkl", 'rb') as file:
        val_dataset_load = pickle.loads(file.read())
    transforms = [  # Easy to define your own transform
        l2l.data.transforms.NWays(val_dataset_load, n=5),
        l2l.data.transforms.KShots(val_dataset_load, k=10),
        l2l.data.transforms.LoadData(val_dataset_load),
    ]
    val_dataset_load = l2l.data.TaskDataset(val_dataset_load, transforms, num_tasks=2000)


    # Create model
    model_ft = models.resnet101(pretrained=None)
    pretrainde_dict = (torch.load('/home/guow/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'))
    newmodel_dict = model_ft.state_dict()
    pretrainde_dict = {k: v for k, v in pretrainde_dict.items() if k in newmodel_dict}
    newmodel_dict.update(pretrainde_dict)
    model_ft.load_state_dict(newmodel_dict)

    # kwargs = {'num_classes':101}
    # model_ft = models.alexnet(pretrained=True, **kwargs)

    # model_ft.fc = nn.Linear(2048, 101)

    model_ft = Share_convs(model_ft, 200)
    # num_ftrs = model_ft.classifier[6].in_features
    # model_ft.classifier[6].out_features = 101

    model = nn.DataParallel(model_ft)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=None)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_data_load.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = val_dataset_load.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
        if iteration%100 == 0  and iteration != 0:
            phase = 'test'
            model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

                    # Iterate over data.
            pbar = tqdm(test_loader)
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                Loss = loss(outputs, labels)
                pbar.set_description("loss %s" % Loss.item())
                running_loss += Loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_description("loss %s" % Loss.item())
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects.double() /len(test_dataset)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    # deep copy the model
            if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'resnet101_v3.pth')

    # meta_test_error = 0.0
    # meta_test_accuracy = 0.0
    # for task in range(meta_batch_size):
    #     # Compute meta-testing loss
    #     learner = maml.clone()
    #     batch = tasksets.test.sample()
    #     evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                        learner,
    #                                                        loss,
    #                                                        adaptation_steps,
    #                                                        shots,
    #                                                        ways,
    #                                                        device)
    #     meta_test_error += evaluation_error.item()
    #     meta_test_accuracy += evaluation_accuracy.item()
    # print('Meta Test Error', meta_test_error / meta_batch_size)
    # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()