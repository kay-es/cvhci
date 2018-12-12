import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
import sys
from model import FeatureResNet, SegResNet
from Strategies import get_path
import matplotlib.pyplot as plt

test = get_path('A1', 'train/img-1153828697.70-0.png') #sys.argv[1]
img = cv2.imread(test, cv2.IMREAD_COLOR).transpose((0, 1, 2))

transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)

imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
plt.show()

input_image=img.unsqueeze(0)
input_image=input_image.type(torch.FloatTensor)
input_image=Variable(input_image)
#input_image=input_image.cuda()

model=torch.load("checkpoints/segrest_150.pt", map_location='cpu')

output_image=model(input_image)

output_image=output_image.squeeze()
#output_image=output_image.cpu()
output_image=output_image.data.numpy()
output_image=output_image.transpose((1,2,0))

imgplot = plt.imshow(output_image)
plt.show()

plt.imsave("output/mask.png", output_image)
#cv2.imwrite('b.png', output_image)