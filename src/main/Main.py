from main.utils.DataLoader import DataLoader
import torch

a1_loader = DataLoader.A1()
a1_train_loader = torch.utils.data.DataLoader(a1_loader.get_train_loader(), batch_size=64, shuffle=True)

for img, label in iter(a1_train_loader):
    print(img)
    print(label)
    print("----")

if False:
    a2_loader = DataLoader.A2()
    a2_train_loader = a2_loader.get_train_loader()

    for img, label in iter(a2_train_loader):
        print(img)
        print(label)
        print("----")
