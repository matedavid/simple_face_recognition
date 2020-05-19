import cv2
import numpy as np

from inference import inference
from feature_extraction import get_detector

IMG_SIZE = (150, 150)

detector = get_detector()

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
    print(f"Image has {len(points)} at start")

    image_features = []

    for pt in points:
        c = False
        if len(image_features) != 0:
            # TODO - Look for better way to approach this 
            for idx, point_c in enumerate(image_features):
                point = point_c[1]
                inside, keep = check_square_inside(pt, point)
                if inside:
                    if keep == 1:

                        del image_features[idx]
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
    
    print(f"Found {len(image_features)} faces...")
    return image_features


    
