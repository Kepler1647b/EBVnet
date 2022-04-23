import torch.nn as nn
from Model.resnet import resnet18, resnet34, resnet50
from Model.vgg import vgg16_bn, vgg19
from efficientnet_pytorch import EfficientNet
from Model.DenseNet import densenet121
import pretrainedmodels


def create_model(model, pretrain):
    if model == 'vgg16_bn':
        Model = vgg16_bn(pretrained=pretrain)
    if model == 'resnet18':
        Model = resnet18(pretrained=pretrain)
    if model == 'resnet50':
        Model = resnet50(pretrained=pretrain)
    if model == 'densenet121':
        Model = densenet121(pretrained=pretrain)
    if model == 'efficientnet-b0':
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=2)
    if model == 'efficientnet-b1':
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b1', num_classes=2)
    if model == 'efficientnet-b5':
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b5', num_classes=2)
    if model == 'se_resnet50':
        Model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
        Model.last_linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    print(model)
    return Model
