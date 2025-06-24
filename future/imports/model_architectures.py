import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_conv=0.3, dropout_fc=0.5):
        """
        Args:
            num_classes (int): número de classes na saída.
            dropout_conv (float): probabilidade de dropout após blocos de conv (opcional).
            dropout_fc (float): probabilidade de dropout nas camadas fully-connected.
        """
        super(SimpleCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_conv)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_conv)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
import torchvision.models as models

class ResnetCNN(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, dropout_fc=0.5):
        """
        Args:
            num_classes (int): número de classes na saída.
            pretrained (bool): se deve carregar pesos pré-treinados da ResNet50.
            dropout_fc (float): probabilidade de dropout nas camadas fully-connected.
        """
        super(ResnetCNN, self).__init__()

        ## tirando o backbone da resnet e deixando pra ele não treinar
        resnet = models.resnet50(pretrained=pretrained)
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) ## tira as ultimas camadas

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = resnet.fc.in_features

        ## parte do classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
