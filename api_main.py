"""
Author=Priyanshu

This file is basically and API which uses images/videos uploaded from your computer and performs the processing 
and returns the results in json format.

"""

import face_recognition
import os
import cv2
import label_image
import label_age_gender
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request, redirect



GENDER_LIST =['Male','Female']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

tf.app.flags.DEFINE_string('model_dir_age', '/home/priyanshu/Downloads/inception/22801', 'Model directory (where training data for AGE lives)')
        
tf.app.flags.DEFINE_string('model_dir_gender', '/home/priyanshu/Downloads/inception/21936', 'Model directory (where training data for GENDER lives)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0', 'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('model_type', 'inception', 'Type of convnet')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

FLAGS = tf.app.flags.FLAGS


config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True


#initializes checkpoints and creates graph for age model
with tf.Graph().as_default() as age_graph:
    sess = tf.Session(graph = age_graph, config=config)       
    age_softmax, age_image = label_age_gender.model_init(sess, FLAGS.model_dir_age, AGE_LIST)
        
 #initializes checkpoints and creates graph for gender model    
with tf.Graph().as_default() as gen_graph:
    sess1 = tf.Session(graph = gen_graph, config=config)
    gender_softmax, gender_image = label_age_gender.model_init(sess1, FLAGS.model_dir_gender, GENDER_LIST)
            

graph = tf.Graph()
graph_def = tf.GraphDef()

with open("retrained_graph.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    target = os.path.join(APP_ROOT, 'uploads/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        return detect_faces_in_image(filename)
    
    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream):
    
    names=[]
    sex=[]
    age=[]
    emotion=[]

    path="datasets"
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
    #file_stream=str(file_stream)
    file_format=file_stream.split(".")

    file_names=os.listdir("uploads")

    for file in file_names:
        if file==file_stream:
            destination = "/".join(["uploads", file])
            video_capture = cv2.VideoCapture(destination) 
    
        

    while True:
    # Grab a single frame of video

        ret, frame = video_capture.read()

        if not ret:
            break
    
        
        if file_format[1]=='jpg' or file_format[1]=='.png' or file_format[1]=='.JPG' or file_format[1]=='.PNG' or file_format[1]=='.jpeg':
            small_frame=frame
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
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
            if not (file_format[1]=='jpg' or file_format[1]=='.png' or file_format[1]=='.JPG' or file_format[1]=='.PNG' or file_format[1]=='.jpeg'):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
            
            
            FaceFileName = "test.jpg" #Saving the current image from the webcam for testing.
            cv2.imwrite(FaceFileName,frame)


            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

            age_gender_data=label_age_gender.main(FaceFileName,sess,sess1,AGE_LIST,GENDER_LIST,age_softmax,age_image,gender_softmax,gender_image)

            gender=str(age_gender_data[2])

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (left + 6, bottom - 6)
            fontScale              = 0.4
            fontColor              = (255,255,255)
            lineType               = 1
            
            text = label_image.main(FaceFileName,graph)# Getting the Result from the label_image file, i.e., Classification Result.
            text = text.title()# Title Case looks Stunning.
                
            #print(name+","+str(age_gender_data[2])+","+str(age_gender_data[0])+"{"+str(age_gender_data[1])+"%"+"}"+",  "+text)
            
            cv2.putText(frame,name+", "+gender[0].title()+", "+str(age_gender_data[0])+"{"+str(age_gender_data[1])+"%"+"}"+", "+text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)        
            # cv2.putText(frame,name+", "+text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            cv2.imwrite("test.jpg",frame)
            names.append(name)
            sex.append(gender)
            age.append(str(age_gender_data[0])+"{"+str(age_gender_data[1])+"%"+"}")
            emotion.append(text)



    result = {
                "Name": names,
                "Sex": sex,
                "Age":age,
                "Emotion": emotion
            }
            
            
    return jsonify(result)

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)