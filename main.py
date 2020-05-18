import click
import cv2
import numpy as np
import pickle as pkl 
import matplotlib.pyplot as plt

import torch 
from inference import inference
from feature_extraction import get_detector

detector = get_detector()

IMG_SIZE = (150, 150)

def transform_image(image, resize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, IMG_SIZE)
    return image

class Square(object):
    def __init__(self, points):
        points_cvt = self.cvt_point(points)
        self.x = (points_cvt[0], points_cvt[2])
        self.y = (points_cvt[1], points_cvt[3])
        
        self.rangx = [i for i in range(self.x[0], self.x[1]+1)]
        self.rangy = [i for i in range(self.y[0], self.y[1]+1)]

    def cvt_point(self, point):
        return [int(round(float(p))) for p in point]

    def __eq__(self, square2):
        if self.x == square2.x and self.y == square2.y:
            return True
        return False

    def area(self):
        return (self.x[1] - self.x[0])**2

    def square_inside(self, square2):
        if (square2.x[0] in self.rangx and square2.x[1] in self.rangx) and \
                (square2.y[0] in self.rangy and square2.y[1] in self.rangy):
            return True
        return False

def check_square_inside(s1, s2):
    # TODO - Look for better way to approach this 
    square1 = Square(s1)
    square2 = Square(s2)

    min_area = min(square1.area(), square2.area())

    little_square = square1 if min_area == square1.area() else square2
    big_square = square1 if little_square == square2 else square2

    if big_square.square_inside(little_square):
        return True, 1 if little_square == square1 else 0
    else:
        return False, -1


def get_features(image_path):
    image = cv2.imread(image_path)
    img = transform_image(image, resize=False)
    
    points = inference(img)
    points = list(points)

    image_features = []

    for pt in points:
        c = False
        if len(image_features) != 0:
            # TODO - Look for better way to approach this 
            for ixd, point_c in enumerate(image_features):
                point = point_c[1]
                inside, keep = check_square_inside(pt, point)
                if inside:
                    if keep == 1:
                        del image_features[ixd]
                    else:
                        c = True
        if c:
            print("Found square inside another, removing...")
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
