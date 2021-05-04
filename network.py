import torch
import torchvision.models as models
from preresnet import resnet

class VGG(torch.nn.Module):
    def __init__(self, vgg='vgg16_bn', data_set='CIFAR10', pretrained=False):
        super(VGG, self).__init__()
        if vgg=='resnet50':
            features = []
            features.append(resnet().conv1())

            features.append(resnet().layer1) # 32x32
            features.append(resnet().layer2)  # 16x16
            features.append(resnet().layer3)  # 8x8
            features.append(resnet().layer4)
            features.append(resnet().bn)
            features.append(resnet().relu)

            features.append(resnet().avgpool)
            self.features = features
        else:
            self.features = models.__dict__[vgg](pretrained=pretrained).features
        
        classifier = []
        if 'CIFAR' in data_set:
            num_class = int(data_set.split("CIFAR")[1])
            
            #classifier.append(torch.nn.Linear(512, 512))
            #classifier.append(torch.nn.BatchNorm1d(512))
            classifier.append(torch.nn.Linear(512, num_class))
        else:
            raise RuntimeError("Not expected data flag !!!")

        if vgg=='resnet50':
            self.classifier = resnet().fc
        self.classifier = torch.nn.Sequential(*classifier)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
