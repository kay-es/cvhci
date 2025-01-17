import random

from torch.utils.data.dataset import Dataset
import os
from skimage import io
import copy
import torchvision
import torch
from torchvision import transforms

import cv2

from pkg_resources import resource_filename


def get_path(task: str, dataset: str, special: str = ""):
    '''
    :param package: the directory to load (e.g. 'A1', 'train' or 'A2', 'test', 'n')
    :return: the relative path of the directory specified by the package path
    '''
    data = dataset + '/' + special if special != "" else dataset
    return 'resources/' + task + '/data/' + data


class Strategy(Dataset):

    def __init__(self, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomCrop, transforms.RandomHorizontalFlip])):
        self.transform = transform

    def get_train_loader(self):
        raise NotImplementedError

    def get_test_loader(self):
        raise NotImplementedError

    def get_validation_loader(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)


class A1(Strategy):

    def __init__(self, transform=transforms.Compose([transforms.ToTensor()])):
        super().__init__(transform)
        self.img_path = None
        self.mask_path = None
        self.data = None

    def get_train_loader(self):
        train_loader = self.copy()
        train_loader.__set_dataset__('train')
        return train_loader

    def get_test_loader(self):
        test_loader = self.copy()
        test_loader.__set_dataset__('test')
        return test_loader

    def get_validation_loader(self):
        validation_loader = self.copy()
        validation_loader.__set_dataset__('validation')
        return validation_loader

    def __set_dataset__(self, dataset):
        self.img_path = get_path('A1', dataset + '/img')
        self.mask_path = get_path('A1', dataset + '/mask')
        self.data = os.listdir(self.img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        im_path = os.path.join(self.img_path, img_name)
        image = cv2.imread(im_path, cv2.IMREAD_COLOR).transpose((0, 1, 2))

        label_name = "mask-" + img_name
        label_path = os.path.join(self.mask_path, label_name)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR).transpose((0, 1, 2))

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            label = self.transform(label)
            normalize = transforms.Normalize((0.24853915,0.266838,0.2138273), (0.16978161, 0.16967748, 0.13661802))
            image = normalize(image)

        return image, label


class A2(Strategy):

    def __init__(self, transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        super().__init__(transform)

    def get_train_loader(self):
        return self.get_dataloader('train')

    def get_test_loader(self):
        return self.get_dataloader('test')

    def get_validation_loader(self):
        return self.get_dataloader('validation')

    def get_dataloader(self, dataset: str):
        dataset = torchvision.datasets.ImageFolder(get_path('A2', dataset), transform=self.transform)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return iterator


class A3(Strategy):

    def __init__(self, transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        super().__init__(transform)

    def get_train_loader(self):
        raise NotImplementedError

    def get_test_loader(self):
        raise NotImplementedError

    def get_validation_loader(self):
        raise NotImplementedError
