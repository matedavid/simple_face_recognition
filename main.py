import click
import cv2
import numpy as np
import pickle as pkl 
import matplotlib.pyplot as plt

import torch 
from inference import inference
from feature_extraction import get_detector


# pred = FaceFeatures()
detector = get_detector()

#IMG_SIZE = (256, 256)
IMG_SIZE = (150, 150)

def transform_image(image, resize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, IMG_SIZE)
    return image


def area_of_square(square):
    return (square[2] - square[0])**2


def check_square_inside(square1, square2):
    # TODO - Look for better way to approach this 
    min_area = min(area_of_square(square1), area_of_square(square2))

    little_square = square1 if min_area == area_of_square(square1) else square2
    big_square = square1 if area_of_square(little_square) == area_of_square(square2) else square2

    top, bottom = False, False

    if little_square[0] >= big_square[0] and little_square[0] <= big_square[2] and little_square[1] >= big_square[1] and little_square[1] <= big_square[3]:
        print("Top left inside")
        top = True
        if little_square[2] >= big_square[0] and little_square[2] <= big_square[2] and little_square[3] >= big_square[1] and little_square[3] <= big_square[3]:
            print("Bottom right inside")
            bottom = True 

    if top and bottom:
        return True, 1 if area_of_square(little_square) == area_of_square(square1) else 0
    else:
        return False, -1

def clean_IoU(image_features):
    pass

def get_features(image_path):
    image = cv2.imread(image_path)
    img = transform_image(image, resize=False)
    
    points = inference(img)
    points = list(points)

    image_features = []

    for pt in points:
        c = False
        if len(image_features) != 0:
            for ixd, point_c in enumerate(image_features):
                point = point_c[1]
                inside, keep = check_square_inside(pt, point)
                if inside:
                    if keep == 1:
                        del image_features[ixd]
                    else:
                        print()
                        c = True
        if c:
            continue

        x1, y1, x2, y2 = pt

        x1 = int(x1)
        y1 = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)

        cropped_image = img[y1:y1+h, x1:x1+w]
        cropped_image = transform_image(cropped_image, resize=True)

        features = detector.compute_face_descriptor(cropped_image)
        features = np.array(features)

        image_features.append([features, pt])

    return image_features
    
def compare(features, target_features):
    return np.linalg.norm(features - target_features)

def show_image_and_identification(image_path, recognition_results):
    img = cv2.imread(image_path)
    img = transform_image(img, resize=False)

    for person_label, points in recognition_results:
        x,y,x2,y2 = points
        img = cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 10)

        plt.imshow(img)
        plt.annotate(person_label, fontsize=15, xy=(.25, .75), xycoords='data', xytext=(x, y))
    
    plt.show()


@click.command()
@click.argument('image_path')
def main(image_path):
    #features, points = get_features(image_path)

    image_features = get_features(image_path)

    with open("people_3dim.pkl", "rb") as f:
        people = pkl.load(f)

    print("Possible people:", list(people.keys()))

    recognition_results = []
    for info in image_features:
        features, points = info

        min_difference = float("inf")
        min_person = ""

        for ppl in people:
            people_features = people[ppl]
            for img_f in people_features:
                diff = compare(features, img_f)

                if diff < min_difference:
                    min_difference = diff
                    min_person = ppl

        if min_difference > 0.6:  # Not sure if this number is good 
            print("No faces recognized")
            #min_person = "Not recognized"
            continue

        print(f"Found: {min_person} {min_difference}")
        recognition_results.append([min_person, points])

    show_image_and_identification(image_path, recognition_results)


if __name__ == "__main__":
    main()
