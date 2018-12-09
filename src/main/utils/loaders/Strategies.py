from main.utils import get_path
from torch.utils.data.dataset import Dataset
import os
from skimage import io
import copy
import torchvision
import torch
from torchvision import transforms

class Strategy(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform

    def get_train(self):
        raise NotImplementedError

    def get_test(self):
        raise NotImplementedError

    def get_validation(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

class A1(Strategy):

    def __init__(self, transform=None):
        super().__init__(transform)
        self.img_path = None
        self.mask_path = None
        self.data = None

    def get_train(self):
        train_iterator = self.copy()
        train_iterator.__set_dataset__('train')
        return train_iterator

    def get_test(self):
        test_iterator = self.copy()
        test_iterator.__set_dataset__('test')
        return test_iterator

    def get_validation(self):
        validation_iterator = self.copy()
        validation_iterator.__set_dataset__('validation')
        return validation_iterator

    def __set_dataset__(self, dataset):
        self.img_path = get_path('A1', dataset + '/img')
        self.mask_path = get_path('A1', dataset + '/mask')
        self.data = os.listdir(self.img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        im_path = os.path.join(self.img_path, img_name)
        image = io.imread(im_path)

        label_name = "mask-" + img_name
        label_path = os.path.join(self.mask_path, label_name)
        label = io.imread(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label

class A2(Strategy):

    def __init__(self, transform=None):
        if transform == None:
            # Define a transform to normalize the data
            default_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        super().__init__(default_transform)

    def get_train(self):
        return self.get_dataset('train')

    def get_test(self):
        return self.get_dataset('test')

    def get_validation(self):
        return self.get_dataset('validation')

    def get_dataset(self, dataset: str):
        dataset = torchvision.datasets.ImageFolder(get_path('A2', dataset), transform=self.transform)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return iterator