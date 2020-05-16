import torch

import cv2
import pickle as pkl 
import os
import numpy as np 
import matplotlib.pyplot as plt
import random

from feature_extraction import FaceFeatures, get_detector
from inference import inference


#IMG_SIZE = (256, 256)
IMG_SIZE = (150, 150)

people = {}

folders = os.listdir("recog_images")

#pred = FaceFeatures()
detector = get_detector()


def transform_image(image, resize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, IMG_SIZE)
    return image

for name in folders:
    folder_path = os.path.join("recog_images", name)
    if not os.path.isdir(folder_path):
        continue

    image_representations = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        print(image_path)

        image = cv2.imread(image_path)
        img = transform_image(image, resize=False)

        points = inference(img)
        points = list(points)

        if len(points) != 1:
            print(f"Image has {len(points)} faces, choosing one randomly")
            # TODO - Implement random choice  
            pts = points[0]
        elif len(points) == 0:
            print("No faces found...")
            continue
        else:
            pts = points[0]
            
        x1, y1, x2, y2 = pts

        x1 = int(x1)
        y1 = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)

        cropped_image = img[y1:y1+h, x1:x1+w]

        """
        plt.imshow(cropped_image)
        plt.show()
        """

        cropped_image = transform_image(cropped_image, resize=True)
        #cropped_image = np.expand_dims(cropped_image, 0)
        #cropped_image = np.rollaxis(cropped_image, 2, 1)

        cropped_image_torch = torch.from_numpy(cropped_image).to(torch.float32)

        """
        features = pred(cropped_image_torch)
        features = features.detach().numpy()
        """

        features = detector.compute_face_descriptor(cropped_image)
        features = np.array(features)
        
        image_representations.append(features)
    
    people[name] = image_representations

with open("people_3dim.pkl", "wb") as f:
    pkl.dump(people, f)