"""
this should only be the model. so data in, process, out. 
this will have to include checkpoints to monitor what is happening. if doing one full image at a time or not
"""

from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the dataloaders
class CustData(Dataset):
    def __init__(self, img_list, knpixval, transform: Optional[bool] = None):
        """
        img_list: one list contains all the annuli of a whole image
        knpixval: knpixvals matching the images.
        """
        self.images = img_list
        self.knpixval = knpixval
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx]), self.knpixval[idx]
        return self.images[idx], self.knpixval[idx]


# define the cnn model here
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # conv layers
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(1, 75, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # downsample (reduce size by half)
            nn.Conv2d(75, 180, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(180, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.LinearLayers = nn.Sequential(
            nn.Linear(64 * 3 * 9, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, 800),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.Linear(400, 200),
            nn.Linear(200, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.ConvLayers(x)
        print(f"x {x.size()}")
        x = x.view(x.size(0), -1)  # dynamicaly flatten the output
        x = self.LinearLayers(x)
        return x
