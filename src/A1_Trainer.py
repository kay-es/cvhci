from argparse import ArgumentParser
import os
import random
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision import models
import re
from DataLoader import DataLoader
from A1_Model import FeatureResNet, SegResNet
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
val_dataset = DataLoader.A1().get_validation_loader()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True)

# Training/Testing
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())

num_classes = 3
net = SegResNet(num_classes, pretrained_net)  # cuda
if torch.cuda.is_available():  # use gpu if available
    net.cuda()

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

# For RMSProp
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if 'bn' in key:
        # No weight decay on batch norm
        params += [{'params': [value], 'weight_decay': 0}]
    elif '.bias' in key:
        # No weight decay plus double learning rate on biases
        params += [{'params': [value], 'lr': 2 * args.lr, 'weight_decay': 0}]
    else:
        params += [{'params': [value]}]

# Define Hyperparams
crit = nn.BCELoss()
if torch.cuda.is_available():
    crit.cuda()
optim = optim.Adam(params, lr=args.lr)
#optim = optim.RMSprop(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=3)

def train(e):
    net.train()
    global checkpoint_iter
    for i, (input, target) in enumerate(train_loader):
        optim.zero_grad()
        if torch.cuda.is_available():
            input, target = Variable(input.cuda()), Variable(target.cuda())  # .cuda(param irgendwas)
        else:
            input, target = Variable(input), Variable(target)
        output = net(input)
        loss = crit(output, target)
        loss.backward()
        optim.step()

        checkpoint_iter += 1
        if i == 0:
            print(f'Epoche: {e}-{(i+1)} - Total: {checkpoint_iter} - Loss: {loss.item()}')
        if checkpoint_iter % 50 == 0:
            torch.save(net, 'output/a1/checkpoints/SegResNet_' + str(checkpoint_iter) + '.pt')
            print(f'Model saved at iteration: {str(checkpoint_iter)}')


for e in range(1, args.epochs + 1):
    train(e)
    #scheduler.step()
