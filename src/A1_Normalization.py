import numpy as np

import torch
from torchvision import transforms
from DataLoader import DataLoader

transform_train = transforms.Compose([
    transforms.ToTensor()
])

trainset = DataLoader.A1().get_train_loader()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=50_000, shuffle=True)

images, labels = iter(train_loader).next()

np_train = images.numpy()

print('Mean: {}'.format(np.mean(np_train, axis=(0, 2, 3))))
# Mean: [0.24853915 0.266838   0.2138273 ]
print('STD: {}'.format(np.std(np_train, axis=(0, 2, 3))))
# STD: [0.16978161 0.16967748 0.13661802]

# PIL IMAGE
# Mean: [-0.00098319  0.2488496   0.550586  ]
# STD: [0.8688914 1.0792519 1.3604592]