import sys
sys.path.append("./utils")
sys.path.append("./architectures")

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import re
import time
from tensorboardX import SummaryWriter

from ConvDeconv import ConvDeconv
from DataLoader import DataLoader

writer = SummaryWriter()
model = ConvDeconv()  # Neural network model object
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
    # os.system('python visualise.py')


criterion = nn.MSELoss()  # Loss Class

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer class
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,gamma=0.1)  # this will decrease the learning rate by factor of 0.1

# https://discuss.pytorch.org/t/can-t-import-torch-optim-lr-scheduler/5138/6
beg = time.time()  # time at the beginning of training
print("Training Started!")
for epoch in range(num_epochs):
    print("\nEPOCH " + str(epoch + 1) + " of " + str(num_epochs) + "\n")

    for img, label in iter(a1_train_loader):
        input = img#.type(torch.FloatTensor)  # typecasting to FloatTensor as it is compatible with CUDA
        output = label#.type(torch.FloatTensor)
        if torch.cuda.is_available():  # move to gpu if available
            input_image = Variable(input.cuda())  # Converting a Torch Tensor to Autograd Variable
            output_image = Variable(output.cuda())
        else:
            input_image = Variable(input)
            output_image = Variable(output)

        optimizer.zero_grad()  # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
        outputs = model(input_image)

        loss = criterion(outputs, output_image)


        loss.backward()  # Backprop
        optimizer.step()  # Weight update
        writer.add_scalar('Training Loss', loss.item() / 10, checkpoint_iter)
        checkpoint_iter = checkpoint_iter + 1

        if checkpoint_iter % 10 == 0 or checkpoint_iter == 1:
            # Calculate Accuracy
            test_loss = 0
            total = 0
            # Iterate through test dataset
            for img_val, mask_val in iter(a1_validation_loader):  # for testing
                input_val = img_val.type(
                    torch.FloatTensor)  # typecasting to FloatTensor as it is compatible with CUDA
                output_val = mask_val.type(torch.FloatTensor)
                if torch.cuda.is_available():  # move to gpu if available
                    input_image_val = Variable(input_val.cuda())  # Converting a Torch Tensor to Autograd Variable
                    output_image_val = Variable(output_val.cuda())
                else:
                    input_image_val = Variable(input_val)
                    output_image_val = Variable(output_val)

                # Forward pass only to get logits/output
                outputs = model(input_image_val)
                test_loss += criterion(outputs, output_image_val).item()
                total += mask_val.size(0)
            test_loss = test_loss / total  # sum of test loss for all test cases/total cases
            writer.add_scalar('Test Loss', test_loss, checkpoint_iter)
            # Print Loss
            time_since_beg = (time.time() - beg) / 60
            print('Iteration: {}. Loss: {}. Test Loss: {}. Time(mins) {}'.format(checkpoint_iter, loss.item() / 10, test_loss,
                                                                                 time_since_beg))
            #oi = outputs[0].squeeze()
            #oi.data = oi.data.type(torch.ByteTensor)
            #oi = oi.data.numpy()
            #oi = oi.transpose((1, 2, 0))

            #imgplot = plt.imshow(oi)
            #plt.show()
        if checkpoint_iter % 50 == 0:
            torch.save(model, 'checkpoints/model_iter_' + str(checkpoint_iter) + '.pt')
            print("model saved at iteration : " + str(checkpoint_iter))
            writer.export_scalars_to_json("graphs/all_scalars_" + str(
                checkpoint_iter_new) + ".json")  # saving loss vs iteration data to be used by visualise.py
    #scheduler.step()
writer.close()