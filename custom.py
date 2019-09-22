# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

#loading the model
json_file = open('fer30.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("fer30.h5")
print("Loaded model from disk")

#setting image resizing parameters
width, height = 48, 48
x, y=None, None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#loading image
img = cv2.imread("happy.jpg")
print("Image Loaded")
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
face = cv2.CascadeClassifier('.\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.32, 5)
print(len(faces))

for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= loaded_model.predict(cropped_img)
        cv2.putText(img, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])

cv2.imshow('Emotion', img)
cv2.waitKey()
