# priyanshu-face_recognition
Real-time face recognition, age, emotion, gender classification using CNN, tensorflow and openCV.

priyanshu-face_recongition can be used to identify and know the emotions, age , gender of a person. It uses your WebCamera and then identifies your characteristics in Real Time.


## PLAN
This is a three step process-

* In the first, we identify the faces using face_recognition API (The world's simplest facial recognition api) 
 https://github.com/ageitgey/face_recognition.

* In second step we retrain our network using inception v3 with our image on four diffrent categories of emotions
  (Happy, Sad, Angry, Calm).

* In third and the last step, we use pretrained checkpoints https://github.com/dpressel/rude-carnie for age and gender classification and then then integrate and set up everything in real time.


## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install face_recognition
```

#### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

### DEPENDENCIES

Hit the following in CMD/Terminal if you don't have already them installed:

    pip install tensorflow
    pip install opencv-python
    pip install flask
    

###  Usage
Command-Line Interface

```
import face_recognition
import cv2
import dlib
import tensorflow
import flask
```

Ensure that following commands run fine. If it does you are good to go.



# STEP 1 - Implementation Using face_recognition for identifying the faces in RealTime

We’re going to use a method invented in 2005 called Histogram of Oriented Gradients — or just HOG for short.

To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces.
Then we’ll look at every single pixel in our image one at a time. For every single pixel, we want to look at the pixels 
that directly surrounding it. Our goal is to figure out how dark the current pixel is compared to the pixels directly 
surrounding it.Then we want to draw an arrow showing in which direction the image is getting darker.If you repeat that 
process for every single pixel in the image, you end up with every pixel being replaced by an arrow.

These arrows are called gradients and they show the flow from light to dark across the entire image.This might seem 
like a random thing to do, but there’s a really good reason for replacing the pixels with gradients. If we analyze pixels 
directly, really dark images and really light images of the same person will have totally different pixel values. But by 
only considering the direction that brightness changes, both really dark images and really bright images will end up with
the same exact representation. That makes the problem a lot easier to solve!


But saving the gradient for every single pixel gives us way too much detail. We end up missing the forest for the trees.
It would be better if we could just see the basic flow of lightness/darkness at a higher level so we could see the basic
pattern of the image. To do this, we’ll break up the image into small squares of 16x16 pixels each. In each square, we’ll 
count up how many gradients point in each major direction (how many point up, point up-right, point right, etc…). Then we’ll
replace that square in the image with the arrow directions that were the strongest.

The end result is we turn the original image into a very simple representation that captures the basic structure of a face 
in a simple way.To find faces in this HOG image, all we have to do is find the part of our image that looks the most similar
to a known HOG pattern that was extracted from a bunch of other training faces:

![known](https://cdn-images-1.medium.com/max/800/1*6xgev0r-qn4oR88FrW6fiA.png)

#### How Face Recognition Works

If you want to learn more about how face location and recognition work instead of depending on a library,read  
[read this article](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

When you install `face_recognition`, you get a two simple command-line 
programs:

* `face_recognition` - Recognize faces in a photograph or folder full for 
   photographs.
* `face_detection` - Find faces in a photograph or folder full for photographs.

#### `face_recognition` command line tool

The `face_recognition` command lets you recognize faces in a photograph or 
folder full  for photographs.

First, you need to put one picture of each person you already know in the (datasets) folder. There should be one
image file for each person with the files named according to who is in the picture:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

#### `face_detection` command line tool

The `face_detection` command lets you find the location (pixel coordinatates) of any faces in an image.

##### Adjusting Tolerance / Sensitivity

If you are getting multiple matches for the same person, it might be that
the people in your photos look very similar and a lower tolerance value
is needed to make face comparisons more strict.

You can do that with the `--tolerance` parameter. The default tolerance
value is 0.6 and lower numbers make face comparisons more strict:

```bash
$ face_recognition --tolerance 0.54 
```

##### Speeding up Face Recognition

Face recognition can be done in parallel if you have a computer with
multiple CPU cores. For example if your system has 4 CPU cores, you can
process about 4 times as many images in the same amount of time by using
all your CPU cores in parallel.

If you are using Python 3.4 or newer, pass in a `--cpus <number_of_cpu_cores_to_use>` parameter:

```bash
$ face_recognition --cpus 4
```

You can also pass in `--cpus -1` to use all CPU cores in your system.

#### Python Module

You can import the `face_recognition` module and then easily manipulate
faces with just a couple of lines of code. It's super easy!

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### Automatically find all the faces in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

You can also opt-in to a somewhat more accurate deep-learning-based face detection model.

Note: GPU acceleration (via nvidia's CUDA library) is required for good
performance with this model. You'll also want to enable CUDA support
when compliling `dlib`.

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations is now an array listing the co-ordinates of each face!
```

##### Recognize faces in images and identify who they are

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
```

## STEP 2 - ReTraining the Network - Emotion Classifier using inception v_3

This step will consist of several sub steps:

- We need to first create a directory named images. In this directory, create five or six sub directories with names like
  Happy, Sad, Angry, Calm and Neutral. You can add more than this.
  
- Now download respective images from the Internet and put it in a separate folder in some other directory. E.g., Download happy images and put it in happy directory(some other location). Remember not to put these downloaded images in directories we created in earlier step, instead put in some other directory anywhere in your pc.
  
 - Now open the "face-crop.py" program.
  - And replace the directory= "location where you have downloaded the photograph"
    For example: ```directory = "C:/Users/user/Downloads/Happy/"```
  
  - Also replace the f_directory="correosponding sub-categories inside the images folder in our repo"
      For example: ```f_directory = "C:/Users/user/priyanshu-face_recognition/images/Happy/"```
  
- Now run the "face-crop.py" program until you fill all the sub-directories inside the images directory wih their 
  respective images. Ensure you put atleast 50-100 images in each subdirectory.
 
 #### Though I have already put the different categories of emotions inside the images directory, but if you want to add more emotions you can use the above steps or use the dataset as it is. It will still give you the result.

  
  
- Once you have only cleaned images, you are ready to retrain the network. For this purpose I'm using inception_v3 Model
  which is quite accurate. To run the training, hit the got to the parent folder and open 
  CMD/Terminal here and hit the following:

```
  python retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --architecture=inception_v3 --image_dir=images
```
- This may take a while depending on your processor and the files named retrained_graph.pb and retrained_labels.txt would be added to your directory.

That's it for this Step.

## STEP 3 - Using Age/Gender classification and then integrating Everything Up

### Currently Supported Models

  - Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis 
    and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, 
    June 2015_

    - http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
    - https://github.com/GilLevi/AgeGenderDeepLearning

  - Inception v3 with fine-tuning
    - This will start with an inception v3 checkpoint, and fine-tune for either age or gender detection 


There are several ways to use a pre-existing checkpoint to do age or gender classification. By default, the code will 
simply assume that the image you provided has a face in it, and will run that image through a multi-pass classification 
using the corners and center.

##### Download the pre-trained age checkpoint for inception from here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWbDZ4Y21GLWxtV1E

##### Download the pre-trained gender checkpoint for inception from here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWemZQd3JMOEZvdGs

- After downloading both the files, unzip them into the inception directory.

### There are different functionalities that are provided in this code:

```api_main.py``` This file is basically an API which uses images/videos uploaded from your computer and performs the processing and returns the results in json format.

```streaming_main.py``` This file is used for giving the results for live streaming where opencv is used for accessing the local camera. It'll open a new window of OpenCV and then identifies your Name, Gender, Age and the emotions.

```url_main.py``` This file is used for processing the stream/image/video from url.


- Open the file you want to execute. Inside the file specify complete path for model_dir_age and model_dir_gender.

For example: ```tf.app.flags.DEFINE_string('model_dir_age', 'C:/Users/user/priyanshu-face_recognition/inception/22801', 'Model directory (where training data for AGE lives)')```

```tf.app.flags.DEFINE_string('model_dir_gender', 'C:/Users/user/priyanshu-face_recognition/inception/21936', 'Model directory (where training data for GENDER lives)')```

##### Finally, you just need to execute it from command line and it will give us the desired results.
 
 For example run the "streaming_main.py" program by typing the following in CMD/Terminal:
 ```
 python streaming_main.py
 ```

It'll open a new window of OpenCV and then identifies your Name, Gender, Age and the emotions.

![known](https://u.imageresize.org/v2/c3a6f5bd-e8ea-4d0c-a0c5-a98276a53d93.jpeg)

## Sources

##### Thanks to ageitgey (https://github.com/ageitgey/face_recognition)

##### Thanks to dpressel (https://github.com/dpressel/rude-carnie)

