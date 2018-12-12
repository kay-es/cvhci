import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
import sys
from main.architectures.ConvDeconv import ConvDeconv
from main.utils.ResourceHelper import get_path

import numpy as np
import matplotlib.pyplot as plt

model=ConvDeconv()
#test = get_path('A1', 'test/img-1153828689.90-0.png') #sys.argv[1]
test = get_path('A1', 'train/img-1153834912.10-0.png') #sys.argv[1]
mask = get_path('A1', 'train/mask-img-1153834912.10-0.png') #sys.argv[1]

img = cv2.imread(test, cv2.IMREAD_COLOR).transpose((0, 1, 2))
mask_img = cv2.imread(mask, cv2.IMREAD_COLOR).transpose((0, 1, 2))


test = cv2.resize(img, (512,512))

transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)
imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
plt.show()
mask_img = transform(mask_img)

imgplot = plt.imshow(mask_img.numpy().transpose((1, 2, 0)))
plt.show()

input_image=img.unsqueeze(0)
input_image=input_image.type(torch.FloatTensor)
input_image=Variable(input_image)


input_mask_image=mask_img.unsqueeze(0)
input_mask_image=input_mask_image.type(torch.FloatTensor)
#imgplot = plt.imshow(input_mask_image.numpy().transpose((1, 2, 0)))
input_mask_image=Variable(input_mask_image)















model=torch.load("checkpoints/model_iter_2000.pt")

output_image=model(input_image)

output_image=output_image.squeeze()
output_image.data=output_image.data.type(torch.ByteTensor)
output_image=output_image.data.numpy()
output_image=output_image.transpose((1,2,0))

imgplot = plt.imshow(output_image)
plt.show()
#r,g,b=cv2.split(output_image)
#output_image=cv2.merge((b,g,r))
# resizing

#display
cv2.imwrite('b.jpg', output_image)