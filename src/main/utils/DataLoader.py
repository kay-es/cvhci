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

    def get_train(self):
        return self.loader.get_train()

    def get_test(self):
        return self.loader.get_test()

    def get_validation(self):
        return self.loader.get_validation()





