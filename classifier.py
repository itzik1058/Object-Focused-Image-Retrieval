import torch
import torch.nn as nn
import torchvision


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.n_features = self.resnet.fc.in_features
        self.out = nn.Linear(self.resnet.fc.in_features, 40)

    def forward(self, x: torch.Tensor, return_features=False):
        self.resnet.eval()
        with torch.no_grad():
            for layer_name, layer in self.resnet._modules.items():
                x = layer(x)
                if layer_name == 'avgpool':
                    break
        features = torch.flatten(x, start_dim=1)
        attr = self.out(features)
        if return_features:
            return features
        return attr

    def predict_np(self, x):
        with torch.no_grad():
            x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
            return self(x).cpu().numpy()
