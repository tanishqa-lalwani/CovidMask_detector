import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import imutils

import itertools
import os 
import cv2
import random 
import glob
import shutil
import matplotlib.pyplot as plt

results={0:'mask',1:'without mask'}

model = load_model('./models/mask_detector.h5')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imagePath = 'C:\projects\dataset_trial\images\images\sample'
os.chdir('C:\projects\dataset_trial\images\images\sample')
rect_size = 4

for c in glob.glob('*.jpg'):

    # test_image = image.load_img(os.path.join(imagePath,c), target_size = (224,224)) 
    # test_image = image.img_to_array(test_image)
    # test_image = test_image.reshape(-1,224,224,3)
    # test_image = np.expand_dims(test_image, axis = -1)

    image = cv2.imread(os.path.join(imagePath,c))
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_img)
    for f in face_coordinates:
        (x, y, w, h) = [v  for v in f] 
        print(x,y,w,h)
        IMG_SIZE=224    
        #img_array=cv2.imread(os.path.join(imagePath,c))
        cropped_img = image[y:y+h, x:x+w]
        new_array=cv2.resize(cropped_img,(IMG_SIZE,IMG_SIZE))
        new_array = new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        reshaped = np.vstack([new_array])
        preds = model.predict(reshaped)

        label=np.argmax(preds,axis=1)[0]

        print(label)

    
        
        #(h, w) = image.shape[:2]

        # startX = int(startX)
        # startY = int(startY)
    
        cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
        cv2.putText(image, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("Output", image)
        key = cv2.waitKey(0)
    



