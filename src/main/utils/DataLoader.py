from main.utils.loaders.Strategies import Strategy, A1, A2

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
        raise NotImplementedError

    def get_train_loader(self):
        return self.loader.get_train_loader()

    def get_test_loader(self):
        return self.loader.get_test_loader()

    def get_validation_loader(self):
        return self.loader.get_validation_loader()





