from main.utils.loaders.Strategies import Strategy, A1

class DataLoader:

    def __init__(self, strategy):
        self.loader: Strategy = strategy

    @staticmethod
    def get_a1_loader():
        return DataLoader(A1())

    def get_train(self):
        return self.loader.get_train()

    def get_test(self):
        return self.loader.get_test()

    def get_validation(self):
        return self.loader.get_validation()



