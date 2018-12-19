from Strategies import Strategy, A1, A2
import numpy as np
from torch.utils.data.dataset import Dataset, Subset
import random

class DataLoader:

    def __init__(self, strategy):
        self.loader: Strategy = strategy

    @staticmethod
    def A1():
        return DataLoader(A1())

    @staticmethod
    def A2():
        return DataLoader(A2())

    @staticmethod
    def A3():
        return DataLoader(A2())

    def get_train_loader(self):
        return self.loader.get_train_loader()

    def get_test_loader(self):
        return self.loader.get_test_loader()

    def get_validation_loader(self):
        return self.loader.get_validation_loader()

    def get_split_set(self):
        ds = self.loader.get_train_loader()
        self.train_valid_split(ds)


class CVSplit():

    def __init__(self, ds: Dataset, valid_split=0.1):
        self.ds: Dataset = ds
        self.valid_split = valid_split

    def get_train_valid_split(self):
        ds_len = len(self.ds)
        valid_size = int(ds_len * self.valid_split)

        valid_indices = random.sample(range(ds_len), valid_size)
        train_indices = list(set(range(ds_len)) - set(valid_indices))

        train_set = Subset(self.ds, train_indices)
        valid_set = Subset(self.ds, valid_indices)

        return train_set, valid_set
