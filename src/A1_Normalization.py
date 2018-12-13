import numpy as np

import torch
from torchvision import transforms
from DataLoader import DataLoader

transform_train = transforms.Compose([
    transforms.ToTensor()
])

trainset = DataLoader.A1().get_train_loader()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=50_000, shuffle=True)

train = train_loader.__iter__().next()[0]

print('Mean: {}'.format(np.mean(train.numpy(), axis=(0, 2, 3))))
# Mean: [0.24853915 0.266838   0.2138273 ]
print('STD: {}'.format(np.std(train.numpy(), axis=(0, 2, 3))))
# STD: [0.16978161 0.16967748 0.13661802]