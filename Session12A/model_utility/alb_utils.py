# from __future__ import print_function
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
import matplotlib.pyplot as plt

import model_utility.data_utils as dutils
import model_utility.model_utils as mutils
import model_utility.plot_utils as putils 
import model_utility.regularization as regularization
import model_utility.alb_utils as alb

import tsai_models.model_cifar as model_cifar
import tsai_models.models as mod

import albumentations as A
from albumentations.pytorch import ToTensor

import grad_cam.grad_cam_viz as grad_cam_viz
import cv2

brightness_val =0.13
cantrast_val = 0.1
saturation_val = 0.10
Random_rotation_val = (-7.0, 7.0) 
fill_val = (1,)

path = os.getcwd()

def get_dataset(train_transforms, test_transforms):
    trainset = datasets.CIFAR10('./', train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10('./', train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataset_img_folder(train_dir, val_dir, train_transforms, test_transforms):
    trainset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    testset = datasets.ImageFolder(root=val_dir, transform=test_transforms)
    return trainset, testset


def find_stats(path):
    mean = []
    stdev = []
    data_transforms = A.Compose([transforms.ToTensor()])
    trainset,testset = get_dataset(data_transforms,data_transforms,path)
    data = np.concatenate([trainset.data,testset.data],axis = 0,out = None)
    data = data.astype(np.float32)/255
    for i in range(data.shape[3]):
        tmp = data[:,:,:,i].ravel()
        mean.append(tmp.mean())
#         mean = [i*255 for i in mean]
        stdev.append(tmp.std())
    print('Image is having {} channels'.format(len(mean)))
    print('Image Means are ',mean)
    print('Image Standard deviations are ',stdev)
    return mean,stdev

class AlbumCompose():
    def __init__(self, transform=None):
        self.transform = transform
        
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

    
# def get_data_transform(path):
#     means, stds = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
#     input_size = 64
#     train_albumentation_transform = A.Compose([
# #                                     A.Cutout(num_holes=3, max_h_size=8, max_w_size=8,  always_apply=True, p=0.7,fill_value=[i*255 for i in mean]),
#                                     A.RandomCrop(always_apply=True, p=0.70),
#                                     A.Rotate( always_apply=True, limit=(-30,30), p=0.70),
#                                     A.HorizontalFlip(p = 0.5,always_apply=True),
#                                     A.HorizontalFlip(always_apply=True, p=0.5),
#                                     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=45, p=0.2),
#                                     A.Normalize(mean=tuple(mean),std=tuple(stdev), max_pixel_value=255,always_apply=True, p=1.0),
#                                     A.Resize(input_size,input_size),
#                                         ToTensor()])

#     # Test Phase transformation
#     test_transforms = transforms.Compose([
#                                           transforms.ToTensor(),
#         transforms.Normalize(tuple(mean),tuple(stdev))
#                                           ])
    
#     train_transforms = AlbumCompose(train_albumentation_transform)
# #     test_transforms = AlbumCompose(test_transforms)
#     return train_transforms, test_transforms

def get_data_transform(path):
    mean, stdev = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    input_size = 64
    train_albumentation_transform = A.Compose([
                                    A.PadIfNeeded (min_height=70, min_width=70,  border_mode=cv2.BORDER_REPLICATE,  always_apply=True, p=1.0),
                                    A.Cutout(num_holes=1, max_h_size=64, max_w_size=64,  always_apply=True, p=0.7,fill_value=[i*255 for i in mean]),
#                                    A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
#                                    A.ChannelShuffle(0.7) ,
                                    A.RandomCrop(height=64,width=64,p=1,always_apply=False),
                                    A.HorizontalFlip(p=0.7, always_apply=True),
                                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=45, p=0.2),
                                    A.Normalize(mean=tuple(mean),std=tuple(stdev), max_pixel_value=255,always_apply=True, p=1.0),
                                    A.Resize(input_size,input_size),
                                        ToTensor()])

    # Test Phase transformation
    test_transforms = transforms.Compose([
                                          transforms.ToTensor(),
        transforms.Normalize(tuple(mean),tuple(stdev))
                                          ])
    train_transforms = AlbumCompose(train_albumentation_transform)
#     test_transforms = AlbumCompose(test_transforms)
    return train_transforms, test_transforms


# Check if cuda is available

def get_device():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    print('Device is',device)
    return device



def get_dataset(train_transforms, test_transforms,path):
    trainset = datasets.CIFAR10(path, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(path, train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataloader(batch_size, num_workers, cuda,path):
    
    print("Running over Cuda !! ", cuda)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_transforms, test_transforms = get_data_transform(path)
    trainset, testset = get_dataset(train_transforms, test_transforms,path)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader,trainset, testset



def get_dataloader_img_folder(train_dir, val_dir, train_transforms, test_transforms, batch_size=32, num_workers=1):
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    trainset, testset = get_dataset_img_folder(train_dir, val_dir, train_transforms, test_transforms)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader
