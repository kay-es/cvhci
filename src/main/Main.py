from main.utils.DataLoader import DataLoader

loader = DataLoader.A1()
train_iterator = loader.get_train()

for img, label in iter(train_iterator):
    print(img)
    print(label)
    print("----")