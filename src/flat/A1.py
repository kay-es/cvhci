import sys
sys.path.append('cv4hci/src')

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import re
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from fcn8 import FCN8

from ConvDeconv import ConvDeconv
from DataLoader import DataLoader

writer = SummaryWriter()
model = FCN8(1)  # Neural network model object
if torch.cuda.is_available():  # use gpu if available
    model.cuda()  # move model to gpu


batch_size = 5 #mini-batch size
a1_loader = DataLoader.A1()
a1_train_dataset = a1_loader.get_train_loader()
a1_train_loader = torch.utils.data.DataLoader(a1_train_dataset, batch_size=batch_size, shuffle=True)

a1_validation_dataset = a1_loader.get_validation_loader()
a1_validation_loader = torch.utils.data.DataLoader(a1_validation_dataset, batch_size=batch_size, shuffle=True)

n_iters = 10000 #total iterations
num_epochs = n_iters / (len(a1_train_dataset) / batch_size)
num_epochs = int(num_epochs)

checkpoint_iter = 0
checkpoint_iter_new = 0
check = os.listdir("checkpoints")  # checking if checkpoints exist to resume training
if len(check):
    check.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
    model = torch.load("checkpoints/" + check[-1])
    checkpoint_iter = int(re.findall(r'\d+', check[-1])[0])
    checkpoint_iter_new = checkpoint_iter
    print("Resuming from iteration " + str(checkpoint_iter))


criterion = nn.MSELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

beg = time.time()
print("Training Started!")
for epoch in range(num_epochs):
    print("\nEPOCH " + str(epoch + 1) + " of " + str(num_epochs) + "\n")

    for input, output in iter(a1_train_loader):
        if torch.cuda.is_available():
            input_image = Variable(input.cuda())
            output_image = Variable(output.cuda())
        else:
            input_image = Variable(input)
            output_image = Variable(output)

        optimizer.zero_grad()
        outputs = model(input_image)
        loss = criterion(outputs, output_image)
        loss.backward()  # Backprop
        optimizer.step()  # Weight update
        writer.add_scalar('Training Loss', loss.item(), checkpoint_iter)
        checkpoint_iter = checkpoint_iter + 1

        if checkpoint_iter % 10 == 0 or checkpoint_iter == 1:
            test_loss = 0
            total = 0
            for input_val, output_val in iter(a1_validation_loader):  # for testing
                if torch.cuda.is_available():  # move to gpu if available
                    input_image_val = Variable(input_val.cuda())  # Converting a Torch Tensor to Autograd Variable
                    output_image_val = Variable(output_val.cuda())
                else:
                    input_image_val = Variable(input_val)
                    output_image_val = Variable(output_val)

                outputs = model(input_image_val)
                test_loss += criterion(outputs, output_image_val).item()
                total += output_val.size(0)
            test_loss = test_loss / total  # sum of test loss for all test cases/total cases

            time_since_beg = (time.time() - beg) / 60
            print(
                'Iteration: {}. Loss: {}. Test Loss: {}. Time(mins) {}'.format(checkpoint_iter, loss.item(), test_loss,
                                                                               time_since_beg))

        if checkpoint_iter % 50 == 0:
            torch.save(model, 'checkpoints/model_iter_' + str(checkpoint_iter) + '.pt')
            print("model saved at iteration : " + str(checkpoint_iter))
