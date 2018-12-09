from main.utils import get_path
from torch.utils.data.dataset import Dataset
import os
from skimage import io

class Strategy(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform

    def get_train(self):
        raise NotImplementedError

    def get_test(self):
        raise NotImplementedError

    def get_validation(self):
        raise NotImplementedError

class A1(Strategy):

    def __init__(self, transform=None):
        super().__init__(transform)
        self.img_path = None
        self.mask_path = None
        self.data = None

    def get_train(self):
        return self.__get_data__('train')

    def get_test(self):
        return self.__get_data__('test')

    def get_validation(self):
        return self.__get_data__('validation')

    def __len__(self):
        return len(self.data)

    def __get_data__(self, dataset):
        self.img_path = get_path('A1', dataset + '/img')
        self.mask_path = get_path('A1', dataset + '/mask')
        self.data = os.listdir(self.img_path)
        return [self.__get_item__(idx) for idx in range(0, self.__len__())]

    def __get_item__(self, idx):
        img_name = self.data[idx]
        im_path = os.path.join(self.img_path, img_name)
        image = io.imread(im_path)

        label_name = "mask-" + img_name
        label_path = os.path.join(self.mask_path, label_name)
        label = io.imread(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label
