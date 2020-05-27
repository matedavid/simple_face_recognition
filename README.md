Simple Face detection and recognition algorithm
--- 

## Resources used:
* detectron2: https://detectron2.readthedocs.io/
* dlib: http://dlib.net/face_recognition.py.html (davisking on Github)

## How to use: 
Unfortunately, the detectron2 model used to recognize images is too big for github, so I haven't been able to include it in the repo. That's why the model should be trained individually in order to have it. 

#### Prepare the models
To train the model, you need to do:
1. Run ```preprocess_data.ipynb```, which will preprocess the image data and prepare it in a pickle file
2. Run ```train.ipynb```, which has all the presets established to imitate my results, so no changes are needed. 

Then, the final model should be saved in a folder called `models`, which should be saved at the level of the main dir. 
Also, you need to download the dlib model for feature extraction of the faces, from this [link](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2). The dlib model should also be saved in the same `models` folder. This model is compressed in a .bz2 format. 

#### Register faces
When the models is set up, you now need to register the faces for the model to recognize.
To do that, you have to create a folder called `recog_images` in the main dir. A folder structure should be created as the following:
```json
-- Recog images (folder)
    - Person 1 name (folder)
        + Image 1 (image)
    - Person 2 name (folder)
        + Image 1 (image)
        + Image 2 (image)
```

You can add multiple images for each person, but it is not needed. 
When you have created all these person models, you can run ```register_recog_images.py``` in order to create a vectorized representation of the images and save them in a file.

#### Run the model 
When the model is set up and the faces registered, the model can be runned to recognize faces on your own images. 
For that, you only need to run 
```bash
python main.py path/to/image
```

Then, in some time, a image with the faces encountered and the person recognized will appear. 