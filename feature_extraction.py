import torch
from torchvision import transforms
import torchvision.models as models

import dlib

"""
alexnet = models.alexnet(pretrained=True)
vgg = models.vgg16(pretrained=True)

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

class FaceFeatures(torch.nn.Module):
    def __init__(self):
        super(FaceFeatures, self).__init__()
        self.features = vgg.features
        
    def forward(self, x):
        x = self.features(x)
        return x
"""

# Code work of: http://dlib.net/face_recognition.py.html
def get_detector():
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    return facerec


if __name__ == "__main__":
    import cv2
    import numpy as np
    img = cv2.imread("test_images/Alba_1.jpg")
    img = cv2.resize(img, (150, 150))
    d = get_detector()
    f = d.compute_face_descriptor(img)
    print(np.array(f).shape)
