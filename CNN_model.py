import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch

class cnn_model(nn.Module):
    def __init__(self, original_model, model_name, bit):
        super(cnn_model, self).__init__()
        if model_name == 'vgg11':
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, bit),
            )
            self.model_name = 'vgg11'
            
        if model_name == 'alexnet':
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, bit),
            )
            self.model_name = 'alexnet'
            
        if model_name == 'resnet18':
            self.features = nn.Sequential(*list(original_model.children())[:-2])  
            self.avgpool = original_model.avgpool
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, bit)
            )
            self.model_name = 'resnet18'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'resnet18':
            f = self.avgpool(f)
            f = torch.flatten(f, 1)
        y = self.classifier(f)
        return y

if __name__=="__main__":
    alexnet = models.alexnet(weights=models.VGG11_Weights.DEFAULT)
    print(alexnet)

