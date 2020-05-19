import pickle as pkl 
import os

from utils import get_features


def register_recog_images():
    people = {}
    folders = os.listdir("recog_images")

    for name in folders:
        folder_path = os.path.join("recog_images", name)
        if not os.path.isdir(folder_path):
            continue

        image_representations = []
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            print(image_path)

            features = get_features(image_path)
            image_representations.append(features[0][0])

        people[name] = image_representations
    return people


if __name__ == "__main__":
    people = register_recog_images()
    with open("people_3dim_prueba.pkl", "wb") as f:
        pkl.dump(people, f)