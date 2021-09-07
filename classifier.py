import torch
import torch.nn as nn
import torchvision


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.n_features = self.resnet.fc.out_features
        self.out = nn.Linear(self.resnet.fc.out_features, 40)

    def forward(self, x: torch.Tensor, return_features=False):
        features = self.resnet(x)
        attr = self.out(features)
        if return_features:
            return features
        return attr
