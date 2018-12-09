from main.utils.DataLoader import DataLoader

a1_loader = DataLoader.A1()
a1_train_iterator = a1_loader.get_train()

for img, label in iter(a1_train_iterator):
    print(img)
    print(label)
    print("----")

a2_loader = DataLoader.A2()
a2_train_iterator = a2_loader.get_train()

for img, label in iter(a2_train_iterator):
    print(img)
    print(label)
    print("----")
