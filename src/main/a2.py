import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Load the training data
trainset = torchvision.datasets.ImageFolder('/src/resource/A2/data/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load the validation data
validationset = torchvision.datasets.ImageFolder('/src/resource/A2/data/validation', transform=transform)
testloader = torch.utils.data.DataLoader(validationset, batch_size=64, shuffle=True)

# Load the test data
testset = torchvision.datasets.ImageFolder('/src/resource/A2/data/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
