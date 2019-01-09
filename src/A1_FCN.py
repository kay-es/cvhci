import torch
from torch import nn
from torchvision import models
import numpy as np
from torchvision.models.resnet import BasicBlock, ResNet


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        self.res = FeatureResNet()
        self.res.load_state_dict(models.resnet34(pretrained=True).state_dict())


        #vgg = models.vgg16()
        #if pretrained:
        #    if caffe:
        #        # load the pretrained vgg16 used by the paper's author
        #        vgg.load_state_dict(torch.load(vgg16_caffe_path))
        #    else:
        #        vgg.load_state_dict(torch.load(vgg16_path))
        #classifier = list(res.classifier.children())


        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        #features[0].padding = (100, 100)

        #for f in features:
        #    if 'MaxPool' in f.__class__.__name__:
        #        f.ceil_mode = True
        #    elif 'ReLU' in f.__class__.__name__:
        #        f.inplace = True

        #self.features3 = nn.Sequential(*features[: 17])
        #self.features4 = nn.Sequential(*features[17: 24])
        #self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        #fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        #fc6.bias.data.copy_(classifier[0].bias.data)
        fc6.weight.data.zero_()
        fc6.bias.data.zero_()
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        #fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        #fc7.bias.data.copy_(classifier[3].bias.data)
        fc7.weight.data.zero_()
        fc7.bias.data.zero_()
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x):

        x1, x2, x3, x4, x5 = self.res(x)

        x_size = x.size()
        #pool3 = self.features3(x)
        #pool4 = self.features4(pool3)
        #pool5 = self.features5(pool4)

        score_fr = self.score_fr(x5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * x5)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * x4)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()