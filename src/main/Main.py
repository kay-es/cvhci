from DataLoader import DataLoader

dataloader = DataLoader.get_a1_loader()
for img, label in dataloader.get_train():
    print(img)