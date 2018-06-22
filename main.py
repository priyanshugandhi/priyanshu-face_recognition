from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import face_recognition
import os
import cv2
import label_image

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import json
import csv

RESIZE_FINAL = 227
GENDER_LIST =['Male','Female']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_string('model_dir_age', '/home/priyanshu/Downloads/inception/22801', 'Model directory (where training data for AGE lives)')
        
tf.app.flags.DEFINE_string('model_dir_gender', '/home/priyanshu/Downloads/inception/21936', 'Model directory (where training data for GENDER lives)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0', 'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint', 'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'inception', 'Type of convnet')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

FLAGS = tf.app.flags.FLAGS

size=4

classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video_capture = cv2.VideoCapture(0)
path="images1"
images_names=os.listdir(path)
known_face_encodings=[]
known_face_names=[]

for imgs in images_names:
    name=imgs.split(".") 
    known_face_names.append(name[0])
    image_path = path + "/" + imgs
    image_file = face_recognition.load_image_file(image_path)
    image_face_encoding = face_recognition.face_encodings(image_file)[0]
    known_face_encodings.append(image_face_encoding)
        




face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    im=cv2.flip(frame,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
            #Save just the rectangle faces in SubRecFaces
            sub_face = im[y:y+h, x:x+w]

            FaceFileName = "test.jpg" #Saving the current image from the webcam for testing.
            cv2.imwrite(FaceFileName, sub_face)
        
            
        def resolve_file(fname):
            if os.path.exists(fname): return fname
            for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
                cand = fname + suffix
                if os.path.exists(cand):
                    return cand
            return None


        def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file):
            try:
                        
                if FLAGS.single_look:
                    image_batch = make_single_crop_batch(image_file, coder)
                else:
                    image_batch = make_multi_crop_batch(image_file, coder)
                
                batch_results = sess.run(softmax_output, feed_dict = {images:image_batch})
                        
                output = batch_results[0]
                batch_sz = batch_results.shape[0]
            
                for i in range(1, batch_sz):
                    output = output + batch_results[i]

                output /= batch_sz
                best = np.argmax(output)
                best_choice = (label_list[best], output[best]*100)
                        
                return best_choice
            
                nlabels = len(label_list)
                if nlabels > 2:
                    output[best] = 0
                    second_best = np.argmax(output)
                    second_best_choice=(label_list[second_best],output[second_best]*100)
                    return second_best_choice
                        
            except Exception as e:
                print(e)
                print('Failed to run image %s ' % image_file)

        def model_init(sess, model_path, label_list): 
            
            model_checkpoint_path, global_step = get_checkpoint(model_path, None, FLAGS.checkpoint)

            nlabels = len(label_list)
            model_fn = select_model(FLAGS.model_type)
            images_placeholder = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            
            with tf.device(FLAGS.device_id):
                    
                logits = model_fn(nlabels, images_placeholder, 1, False)
                    
                init = tf.global_variables_initializer()
                                   
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                                    
                softmax_output = tf.nn.softmax(logits)
                                
                return softmax_output, images_placeholder  

        def m(filename):

            files = []

            if (os.path.isdir(filename)):
                for relpath in os.listdir(filename):
                    abspath = os.path.join(filename, relpath)
                    
                    if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                        print(abspath)
                        files.append(abspath)
            else:
                files.append(filename)
                # If it happens to be a list file, read the list and clobber the files
                if any([filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                    files = list_images(filename)
            
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True

            #initializes checkpoints and creates graph for age model
            with tf.Graph().as_default() as age_graph:
                sess = tf.Session(graph = age_graph, config=config)       
                age_softmax, age_image = model_init(sess, FLAGS.model_dir_age, AGE_LIST)
            #initializes checkpoints and creates graph for gender model    
            with tf.Graph().as_default() as gen_graph:
                sess1 = tf.Session(graph = gen_graph, config=config)
                gender_softmax, gender_image = model_init(sess1, FLAGS.model_dir_gender, GENDER_LIST)
            
            coder_age = ImageCoder()
            coder_gender = ImageCoder()

            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            
            for image_file in image_files:
                best_data,second_best_data=classify_one_multi_crop(sess, AGE_LIST, age_softmax, coder_age, age_image, image_file)
                gender_data=classify_one_multi_crop(sess1, GENDER_LIST, gender_softmax, coder_gender, gender_image, image_file)
            
            age_gender_data=[]
            age_gender_data.append(best_data)
            age_gender_data.append(int(second_best_data))
            age_gender_data.append(gender_data[0])
            return age_gender_data
            sess.close()
            sess1.close()
                

        age_gender_data=m(FaceFileName)

        gender=str(age_gender_data[2])


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (left + 6, bottom - 6)
        fontScale              = 0.4
        fontColor              = (255,255,255)
        lineType               = 1
        text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
        text = text.title()# Title Case looks Stunning.
        print(name+","+str(age_gender_data[2])+","+str(age_gender_data[0])+"{"+str(age_gender_data[1])+"%"+"}"+",  "+text)
        cv2.putText(frame,name+", "+gender[0].title()+", "+str(age_gender_data[0])+"{"+str(age_gender_data[1])+"%"+"}"+", "+text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


