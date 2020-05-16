import click
import cv2
import numpy as np
import pickle as pkl 
import matplotlib.pyplot as plt

import torch 
from inference import inference
from feature_extraction import FaceFeatures


pred = FaceFeatures()
IMG_SIZE = (256, 256)

def transform_image(image, resize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, IMG_SIZE)
    return image

def get_features(image_path):
    image = cv2.imread(image_path)
    img = transform_image(image, resize=False)
    
    points = inference(img)
    points = list(points)

    if len(points) != 1:
        print(f"Image has {len(points)} faces, choosing one randomly")
        # TODO - Implement random choice
        ridx = np.random.randint(0, len(points))
        pts = points[ridx]
    else:
        pts = points[0]

    x1, y1, x2, y2 = pts

    x1 = int(x1)
    y1 = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)

    cropped_image = img[y1:y1+h, x1:x1+w]

    cropped_image = transform_image(cropped_image, resize=True)
    cropped_image = np.expand_dims(cropped_image, 0)
    cropped_image = np.rollaxis(cropped_image, 3, 1)

    cropped_image_torch = torch.from_numpy(cropped_image).to(torch.float32)

    features = pred(cropped_image_torch)
    features = features.detach().numpy()
    #return np.array(features), points
    return features, pts
    
def compare(features, target_features):
    # ‖f(x1)−f(x2)‖_2^2
    """
    d1 = np.linalg.norm(features[0])
    d2 = np.linalg.norm(target_features[0])
    """
    return np.linalg.norm(features[0] - target_features[0])

def show_image_and_identification(image_path, points, person_label):
    img = cv2.imread(image_path)
    img = transform_image(img, resize=False)

    x,y,x2,y2 = points
    img = cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 10)

    plt.imshow(img)
    plt.annotate(person_label, fontsize=20, xy=(.25, .75), xycoords='data', xytext=(x, y))
    plt.show()


@click.command()
@click.argument('image_path')
def main(image_path):
    features, points = get_features(image_path)

    with open("people_3dim.pkl", "rb") as f:
        people = pkl.load(f)    

    min_difference = float("inf")
    min_person = ""
    for ppl in people:
        people_features = people[ppl]
        for img in people_features:
            d1 = img.detach().numpy()
            diff = compare(features, d1)
            
            if diff < min_difference:
                min_difference = diff
                min_person = ppl
    
    show_image_and_identification(image_path, points, min_person)
    print(min_person, min_difference)



if __name__ == "__main__":
    main()
