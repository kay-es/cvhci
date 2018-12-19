from argparse import ArgumentParser
import os
import random
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision import models
import re
from DataLoader import DataLoader, CVSplit
from A1_Model import FeatureResNet, SegResNet
import copy
#from A1_FCN import FCN8s
import matplotlib.pyplot as plt

# Setup
parser = ArgumentParser(description='Semantic segmentation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=3, help='Data loader workers')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--crop-size', type=int, default=512, help='Training crop size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.2, help='Momentum')
parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay')
parser.add_argument('--batch-size', type=int, default=6, help='Batch size')

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

# https://github.com/Kaixhin/FCN-semantic-segmentation

# Data
train_dataset = DataLoader.A1().get_train_loader()
#val_dataset = DataLoader.A1().get_validation_loader()
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                                           num_workers=args.workers, pin_memory=True)
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
#                                         pin_memory=True)


cv_splitter = CVSplit(train_dataset, 0.1)

# Training/Testing
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
num_classes = 3 #RGB?
net = SegResNet(num_classes, pretrained_net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():  # use gpu if available
    net.to(device)

# Load current state of model
checkpoint_iter = 0
if not os.path.exists('output/a1/checkpoints'):
    os.makedirs('output/a1/checkpoints')
check = os.listdir("output/a1/checkpoints")
if len(check):
    check.sort(key=lambda x: int((x.split('_')[1]).split('.')[0]))
    if torch.cuda.is_available():
        net = torch.load("output/a1/checkpoints/" + check[-1])
    else:
        net = torch.load("output/a1/checkpoints/" + check[-1], map_location='cpu')
    checkpoint_iter = int(re.findall(r'\d+', check[-1])[0]) + 1
    print("Resuming from iteration " + str(checkpoint_iter))

# Define Hyperparams
crit = nn.BCELoss()
if torch.cuda.is_available():
    crit.cuda()
optim = optim.Adam(net.parameters(), lr=args.lr)

# TRAIN METHOD
def train(e, train_loader, valid_loader):
    print("LEN TRAIN:", len(train_loader))
    print("LEN VALID:", len(valid_loader))
    loaders = dict({'train': train_loader, 'val': valid_loader})
    global checkpoint_iter

    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for i, (input, target) in enumerate(loaders[phase]):
            optim.zero_grad()

            input, target = input.to(device), target.to(device)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(input)
                _, preds = torch.max(outputs, 1)
                loss = crit(outputs, target)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optim.step()

            # OUTPUT ONLY
            checkpoint_iter += 1
            #if i == 0:
             #   print(f'Epoche: {e}-{(i+1)} - Total: {checkpoint_iter} - Loss: {loss.item()}')
            if checkpoint_iter % 50 == 0:
                torch.save(net, 'output/a1/checkpoints/SegResNet_' + str(checkpoint_iter) + '.pt')


        epoch_loss = running_loss / len(loaders[phase])
        epoch_acc = running_corrects.double() / len(loaders[phase])

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())

    print()



# TRAIN
for epoch in range(args.epochs):
    print('Epoch {}/{}'.format(epoch, args.epochs - 1))
    print('-' * 10)

    train_set, valid_set = cv_splitter.get_train_valid_split()
    train_loader  = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    train(epoch, train_loader, valid_loader)
