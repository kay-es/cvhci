import skorch
import torch
from A1_Model import FeatureResNet, SegResNet
from torchvision import models
import numpy as np

n_classes = 12

pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())

num_classes = 3
network = SegResNet(num_classes, pretrained_net)
net = skorch.NeuralNet(
    module=network,
    criterion=torch.nn.BCELoss,
    use_cuda=True,
    batch_size=5,
)

params = {
    'lr': [0.01, 0.02],
    'max_epochs': [5, 10]
}

# if only training
# net.fit(X=X, y=y)

image_indicators = np.hstack([np.repeat(i, len(x)) for i, x in
                                      enumerate(X)])
labels = image_indicators % n_classes
X, y = np.vstack(X), np.hstack(Y)

cv = LeavePLabelOut(labels=labels, p=1)
gs = GridSearchCV(net, params, scoring='f1', verbose=10, cv=cv, n_jobs=-1)

gs.fit(X=X, y=y)
print(gs.best_score_, gs.best_params_)