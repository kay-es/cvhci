import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
from Strategies import get_path
import matplotlib.pyplot as plt
from DataLoader import DataLoader
import numpy as np
import PIL
from matplotlib import cm
from PIL import Image

# train_dataset = DataLoader.A1().get_train_loader()
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
#                                            num_workers=1, pin_memory=True)
# transform = transforms.Compose([transforms.ToTensor()])
# normalize = transforms.Normalize((-3.3960785e-07,-2.2124875e-06,1.1708015e-06), (0.9999996, 0.99999946, 1.0000006))
# #img = cv2.imread(get_path('A1', 'train/img-1153828697.70-0.png'), cv2.IMREAD_COLOR).transpose((0, 1, 2))
# img = PIL.Image.open(get_path('A1', 'train/img-1153828697.70-0.png')).convert('RGB')
# img = transform(img)
# imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
# plt.show()
#
# #img = normalize(img)
# imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
# plt.show()
#
# input_image = img.unsqueeze(0)
# input_image = input_image.type(torch.FloatTensor)
# if torch.cuda.is_available():
#     input_image = input_image.cuda()
# input_image = Variable(input_image)
# model_name = "SegResNet_750"
# model = torch.load("output/a1/checkpoints_/" + model_name + ".pt", map_location='cpu')
# output_image = model(input_image)
# output_image = output_image.squeeze()
# if torch.cuda.is_available():
#     output_image = output_image.cpu()
# output_image = output_image.data.numpy()
# output_image = output_image.transpose((1, 2, 0))
#
# # Transform to B/W
# output_image = (output_image > 0.5).astype(float)
# imgplot = plt.imshow(output_image)
# plt.show()
#
# for img, label in train_loader:
#     np_img = np.squeeze(img.numpy(), axis=0)
#     imgplot = plt.imshow(np_img.transpose((1, 2, 0)))
#     plt.show()
#     np_label = np.squeeze(label.numpy(), axis=0)
#     imgplot = plt.imshow(np_label.transpose((1, 2, 0)))
#     plt.show()


#

test_dir = os.listdir(get_path('A1', 'test'))
model_name = "SegResNet_8007.1545308275.99"
model = torch.load("output/a1/checkpoints_/" + model_name + ".pt", map_location='cpu')
transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
if not os.path.exists('output/a1/processed/' + model_name):
    os.makedirs('output/a1/processed/' + model_name)
for i, test in enumerate(test_dir):
    img = PIL.Image.open(get_path('A1', 'test/' + test)).convert('RGB')
    #img = cv2.imread(get_path('A1', 'test/' + test), cv2.IMREAD_COLOR).transpose((0, 1, 2))
    img = transform(img)
    # imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
    # plt.show()

    input_image = img.unsqueeze(0)
    input_image = input_image.type(torch.FloatTensor)
    if torch.cuda.is_available():
        input_image = input_image.cuda()
    input_image = Variable(input_image)

    output_image = model(input_image)


    output_image = output_image.squeeze()
    if torch.cuda.is_available():
        output_image = output_image.cpu()


    #output_image = torch.FloatTensor(3, 480, 640)
    #output_image = transforms.ToPILImage()(output_image)
    #plt.imshow(output_image)
    #plt.show()
    #output_image.save("output/a1/processed/" + model_name + "/out-" + test, "png")
    #output_image = output_image.data.numpy()
    #output_image = output_image.transpose((1, 2, 0))


    # Transform to B/W
    #output_image = (output_image > 0.1).astype(float)

    #output_image = Image.fromarray(output_image.astype('uint8') * 255, 'RGB')

    detransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])
    output_image = detransform(output_image)

    output_image = output_image.data.numpy()
    output_image = output_image.transpose((1, 2, 0))
    output_image = np.uint8((output_image > 0.025)*255)


    to_ten = transforms.Compose([
        transforms.ToTensor()
    ])

    output_image = to_ten(output_image)

    to_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])

    output_image = to_pil(output_image)

    output_image = output_image.convert('RGB')
    output_image.save("output/a1/processed/" + model_name + "/out-" + test)


    #output_image = output_image.transpose((1, 2, 0))
    #output_image = (output_image > 0.15).astype(float)

    #imgplot = plt.imshow(output_image)
    # plt.show()
    print(i, "of", len(test_dir) - 1, test)

    #plt.imsave("output/a1/processed/" + model_name + "/out-" + test, output_image)





#
# normalize = transforms.Normalize((-3.3960785e-07,-2.2124875e-06,1.1708015e-06), (0.9999996, 0.99999946, 1.0000006))
# img = cv2.imread(get_path('A1', 'train/img-1153828697.70-0.png'), cv2.IMREAD_COLOR).transpose((0, 1, 2))
# img = transform(img)
# imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
# plt.show()
#
# img = normalize(img)
# imgplot = plt.imshow(img.numpy().transpose((1, 2, 0)))
# plt.show()
#
# input_image = img.unsqueeze(0)
# input_image = input_image.type(torch.FloatTensor)
# if torch.cuda.is_available():
#     input_image = input_image.cuda()
# input_image = Variable(input_image)
#
# output_image = model(input_image)
# output_image = output_image.squeeze()
# if torch.cuda.is_available():
#     output_image = output_image.cpu()
# output_image = output_image.data.numpy()
# output_image = output_image.transpose((1, 2, 0))
#
# # Transform to B/W
# output_image = (output_image > 0.1).astype(float)
# imgplot = plt.imshow(output_image)
# plt.show()
