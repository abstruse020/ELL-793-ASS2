# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 00:06:02 2022

@author: bokar
"""
import torch
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms as T


#%% Getting Data

transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#%%



#%%
resnet18 = models.resnet18(pretrained=False)
print(resnet18)

# Input size of resnet18 -> 224 by 224
