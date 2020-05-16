import torch
from torchvision import transforms
import torchvision.models as models

import numpy as np
import cv2
import matplotlib.pyplot as plt

alexnet = models.alexnet(pretrained=True)

class FaceFeatures(torch.nn.Module):
    def __init__(self):
        super(FaceFeatures, self).__init__()
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = torch.nn.Sequential(*list(alexnet.classifier)[:6])
        
    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)
        return x