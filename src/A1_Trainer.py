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
#from A1_Model import FeatureResNet, SegResNet
from A1_ModelDeeper import FeatureResNet, SegResNet
import copy
import time
from sklearn.metrics import f1_score
import numpy as np
from A1_PSP import PSPNet

#from A1_FCN import FCN8s
import matplotlib.pyplot as plt

# Setup
parser = ArgumentParser(description='Semantic segmentation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=3, help='Data loader workers')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--crop-size', type=int, default=512, help='Training crop size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.2, help='Momentum')
parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay')
parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--batch-size-val', type=int, default=2, help='Val Batch size')
parser.add_argument('--model', type=str, default="segresnet34", help='Val Batch size')

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

# https://github.com/Kaixhin/FCN-semantic-segmentation

# Data
train_dataset = DataLoader.A1().get_train_loader()
valid_dataset = DataLoader.A1().get_validation_loader()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size_val, num_workers=args.workers,
                                         pin_memory=True)


cv_splitter = CVSplit(train_dataset, 0.15)

# Training/Testing
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet50(pretrained=True).state_dict())
num_classes = 3 #RGB?
net = SegResNet(num_classes, pretrained_net)
#net = PSPNet(num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if torch.cuda.is_available():  # use gpu if available
#    net.cuda()
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

since = time.time()
best_model_wts = copy.deepcopy(net.state_dict())
best_f1 = 0.0


# TRAIN METHOD
def train(e, train_loader, valid_loader):
    loaders = dict({'train': train_loader, 'val': valid_loader})
    global checkpoint_iter
    global best_model_wts
    global best_f1

    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        for i, (input, target) in enumerate(loaders[phase]):
            optim.zero_grad()

            input, target = Variable(input.to(device)), Variable(target.to(device))

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(input)
                #_, preds = torch.max(outputs)
                loss = crit(outputs, target)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optim.step()

            # OUTPUT ONLY
            checkpoint_iter += 1
            #if i == 0:
             #   print(f'Epoche: {e}-{(i+1)} - Total: {checkpoint_iter} - Loss: {loss.item()}')
            #if checkpoint_iter % 50 == 0:
             #   torch.save(net, 'output/a1/checkpoints/SegResNet_' + str(checkpoint_iter) + '.pt')

            # statistics
            running_loss += loss.item()

            y_p = outputs.view(6, -1)
            y_t = target.view(6, -1)
            y_p = np.array(y_p.cpu().data) > 0.33
            y_t = np.array(y_t.cpu().data) > 0.33

            f1 = f1_score(y_t, y_p, average="samples")
            running_corrects += f1

        epoch_loss = running_loss / len(loaders[phase])
        epoch_f1 = running_corrects / len(loaders[phase])

        print('{} \tLoss: {:.4f} F1: {:.4f}'.format(
            phase, epoch_loss, epoch_f1))

        # deep copy the model
        if phase == 'val' and epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_model_wts = copy.deepcopy(net.state_dict())

    print()



#train_set, valid_set = cv_splitter.get_train_valid_split()
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
#                                           num_workers=args.workers, pin_memory=True)
#valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True,
#                                           num_workers=args.workers, pin_memory=True)

# TRAIN
for epoch in range(args.epochs):
    print('Epoch {}/{}'.format(epoch, args.epochs - 1))
    print('-' * 10)
    train(epoch, train_loader, valid_loader)
    net.load_state_dict(best_model_wts)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val F1: {:4f}'.format(best_f1))

# load best model weights

torch.save(net, 'output/a1/checkpoints/SegResNet_' + str(checkpoint_iter) + "." + str(time.time()) + '.pt')