import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.datasets 
import torchvision.datasets 
import torchvision.transforms
from skimage import io, transform

class DatasetA1(Dataset):

    def __init__(self, img_path, mask_path, transform=None):
            self.images = os.listdir(img_path)
            self.img_path = img_path
            self.mask_path = mask_path
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_name = self.images[idx]
        im_path = os.path.join(self.img_path, img_name)
        image = io.imread(im_path)

        label_name =  "mask-" + img_name
        label_path =  os.path.join(self.mask_path, label_name)
        label = io.imread(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label