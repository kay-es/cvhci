import torch
import os
from torch.autograd import Variable
from torchvision import transforms
import cv2
from Strategies import get_path
import matplotlib.pyplot as plt

test_dir = os.listdir(get_path('A1', 'test'))
model_name = "SegResNet_2400"
model = torch.load("output/a1/checkpoints/" + model_name + ".pt", map_location='cpu')
transform = transforms.Compose([transforms.ToTensor()])
if not os.path.exists('output/a1/processed/' + model_name):
    os.makedirs('output/a1/processed/' + model_name)
for i, test in enumerate(test_dir):
    img = cv2.imread(get_path('A1', 'test/' + test), cv2.IMREAD_COLOR).transpose((0, 1, 2))
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
    output_image = output_image.data.numpy()
    output_image = output_image.transpose((1, 2, 0))

    # Transform to B/W
    processed_output = (output_image > 0.4).astype(float)
    # imgplot = plt.imshow(processed_output)
    # plt.show()
    print(i, "of", len(test_dir) - 1, test)

    plt.imsave("output/a1/processed/" + model_name + "out-" + test, processed_output)
