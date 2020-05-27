Simple Face detection and recognition algorithm
---

## Resources used:
* detectron2: https://detectron2.readthedocs.io/
* dlib: http://dlib.net/face_recognition.py.html (davisking on Github)

## How to use: 
Unfortunately, the detectron2 model used to find faces in an image is too big for github, so it can't be included in the repo. That's why the model has to be individually in order to have it. 

#### Prepare the models
To train the model, you need to do:
1. Run ```preprocess_data.ipynb``` (in a jupyter notebook), which will preprocess the image data and prepare it in a pickle file for training. 
2. Run ```train.ipynb``` (in a jupyter notebook). It has all the presets established to imitate my results, so no changes are needed.

Then, the final model should be saved in a folder called `models/`. 
Also, you need to download the **dlib** model for feature extraction of the faces, from this [link](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2). The dlib model should also be saved in the same `models/` folder. This file is compressed in a .bz2 format, so remember to decompress it before. 

#### Register faces
When the models are set up, you need to register the faces that you want to model to recognize. 
To do that, you have to create a folder called `recog_images/`. A folder structure should be created as the following:
```json
-- Recog images (folder)
    - Person 1 name (folder)
        + Image 1 (image)
    - Person 2 name (folder)
        + Image 1 (image)
        + Image 2 (image)
```

You can add multiple images for each person, but it's not needed. 
When you added all the persons and corresponding images, you can run ```register_recog_images.py``` in order to create a vectorized representation of the images and save them to a file. 

#### Run the model 
When all the previous requirements are met, you can now use the algorithm on your own images. 
For that, you only need to run:
```bash
python main.py path/to/your_own_image
```

Then, an image with the faces detected and recognized should appear. 


## Samples: 
##### models/ and recog_images/ structure
![Recog images structure](https://github.com/matedavid/simple_face_recognition/blob/master/.github/Recog_images%20structure_example_1.png)
##### Results example
![Algorithm in work example](https://github.com/matedavid/simple_face_recognition/blob/master/.github/Result_1_elon-jack.png)