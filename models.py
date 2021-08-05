import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from scipy.io import loadmat
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SRRED(nn.Module):
    def __init__(self, in_channels, p=0.4):
        super(SRRED, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        l = len(list(resnet50.children()))
        self.resnet = nn.Sequential(*(list(resnet50.children())[1:l-1]))
        self.stem = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=(2,2), padding=(3,3), bias=False)
        self.linear1 = nn.Linear(resnet50.fc.in_features,512)
        self.linear2 = nn.Linear(512,1)
        self.LReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out_stem = self.stem(x)
        out_resnet = self.resnet(out_stem)
        out_linear1 = self.LReLU(self.linear1(out_resnet.view(out_resnet.shape[0],-1)))
        out = self.ReLU(self.linear2(out_linear1))
        return out

class SRRED_Dataset(Dataset):
    def __init__(self,contents,labels,transform=None):
        self.transform = transform
        self.imgs = []
        for ii, currName in enumerate(contents):
            self.imgs.append((currName, labels[ii]))

    def __getitem__(self, index):
        curr_img_path,curr_label = self.imgs[index]
        curr_data = loadmat(curr_img_path)
        tmp_img = curr_data["panelImage"].astype(np.uint8)
        if not (self.transform):
            self.transform = T.Compose([T.ToTensor(),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        transforms.Normalize(mean=[0.485], std=[0.229])
                                        ])
        curr_img = self.transform(tmp_img)
        curr_label = torch.tensor(curr_label)
        return curr_img, curr_label.float()
    def __len__(self):
        return len(self.imgs)

class TRRED(nn.Module):
    def __init__(self, in_channels, p=0.4):
        super(TRRED, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        l = len(list(resnet50.children()))
        self.resnet = nn.Sequential(*(list(resnet50.children())[1:l-1]))
        self.stem = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=(2,2), padding=(3,3), bias=False)
        self.linear1 = nn.Linear(resnet50.fc.in_features,512)
        self.linear2 = nn.Linear(512,1)
        self.LReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out_stem = self.stem(x)
        out_resnet = self.resnet(out_stem)
        out_linear1 = self.LReLU(self.linear1(out_resnet.view(out_resnet.shape[0],-1)))
        out = self.ReLU(self.linear2(out_linear1))
        return out

class TRRED_Dataset(Dataset):
    def __init__(self,contents,labels,transform=None):
        self.transform = transform
        self.imgs = []
        for ii, currName in enumerate(contents):
            self.imgs.append((currName, labels[ii]))

    def __getitem__(self, index):
        curr_img_path,curr_label = self.imgs[index]
        curr_data = loadmat(curr_img_path)
        tmp_img = curr_data["panelImage"].astype(np.uint8)
        if not (self.transform):
            self.transform = T.Compose([T.ToTensor(),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        transforms.Normalize(mean=[0.485], std=[0.229])
                                        ])
        curr_img = self.transform(tmp_img)
        curr_label = torch.tensor(curr_label)
        return curr_img, curr_label.float()
    def __len__(self):
        return len(self.imgs)
