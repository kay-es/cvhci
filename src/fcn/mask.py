import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
import sys
from model import FeatureResNet, SegResNet
from Strategies import get_path
import matplotlib.pyplot as plt
import numpy as np


test_path = os.listdir(get_path('A1', 'test'))
model=torch.load("checkpoints/segrest_1350.pt", map_location='cpu')

for i, test in enumerate(test_path):
    #test = get_path('A1', 'test/img-1153828689.90-0.png') #sys.argv[1]
    img = cv2.imread(get_path('A1', 'test/' + test), cv2.IMREAD_COLOR).transpose((0, 1, 2))

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    #imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
    #plt.show()

    input_image=img.unsqueeze(0)
    input_image=input_image.type(torch.FloatTensor)
    input_image=Variable(input_image)
    #input_image=input_image.cuda()


    output_image=model(input_image)

    output_image=output_image.squeeze()
    #output_image=output_image.cpu()
    output_image=output_image.data.numpy()
    output_image=output_image.transpose((1,2,0))



    processed_output = (output_image > 0.4).astype(float)
    print(i, "of", len(test_path), test)


    #imgplot = plt.imshow(processed_output)
    #plt.show()

    plt.imsave("output/out-" + test, processed_output)
    #cv2.imwrite('b.png', output_image)