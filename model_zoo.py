import torch
torch.hub.set_dir(r'E:\Torch_hub')
import torch.nn as nn
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_create(model_name, dataset_name, pretrained=True):
    model = getattr(models, model_name)(pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if dataset_name == 'CIFAR-100':
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif dataset_name == 'CIFAR-10':
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif dataset_name == 'Tiny-ImageNet':
        model.fc = nn.Linear(model.fc.in_features, 200)
    else:
        raise NotImplementedError('Dataset not implemented')
    return model