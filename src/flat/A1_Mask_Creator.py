import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
import sys
from ConvDeconv import ConvDeconv
from ResourceHelper import get_path
import matplotlib.pyplot as plt

model=ConvDeconv()
test = get_path('A1', 'test/img-1153828689.90-0.png') #sys.argv[1]
img = cv2.imread(test, cv2.IMREAD_COLOR).transpose((0, 1, 2))



transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)

imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
plt.show()

input_image=img.unsqueeze(0)
input_image=input_image.type(torch.FloatTensor)
input_image=Variable(input_image)
#input_image=input_image.cuda()

model=torch.load("checkpoints/model_iter_2000.pt", map_location='cpu')

output_image=model(input_image)

output_image=output_image.squeeze()
#output_image=output_image.cpu()
#output_image.data=output_image.data.type(torch.ByteTensor)
output_image=output_image.data.numpy()
output_image=output_image.transpose((1,2,0))
#r,g,b=cv2.split(output_image)
#output_image=cv2.merge((b,g,r))
# resizing

imgplot = plt.imshow(output_image)
plt.show()

plt.imsave("onePic.png", output_image)
#display
#cv2.imwrite('b.png', output_image)