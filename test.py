import numpy as np
import os
from PIL import Image

TRAIN_DATA= 'datasets/train_data'
TEST_DATA='datasets/test_data'

xtrain=[]
ytrain=[]

xtest=[]
ytest=[]

dict= {'Cuong':[1,0,0,0], 'duc':[0,1,0,0], 'Vinh':[0,0,1,0], 'Đản':[0,0,0,1]}

def getdata(dirdata, listdata):
    for whatever in os.listdir(dirdata):
        whatever_path = os.path.join(dirdata,whatever)
        list_filename_path= []
        for filenames in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path,filenames)
            label= filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            list_filename_path.append((img, dict[label]))

        listdata.extend(list_filename_path)
    return listdata

xtrain = getdata(TRAIN_DATA, xtrain)
xtest = getdata(TEST_DATA, xtest )

print(xtrain[500])

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras import models
import tensorflow as tf

model_training = models.Sequential([
    #
    layers.Conv2D(32,(3,3), input_shape=(100,100, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(64,(3, 3),  activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(), # 32x32, 3 layers
    layers.Dense(100,activation= 'relu'),  #fully connective
    layers.Dense(50,activation= 'relu'),
    layers.Dense(4,activation='softmax')
])

#
# model_training.compile(optimizer= 'adam', # update parameter
#                        loss='categorical_crossentropy', #tinh mat mat
#                        metrics=["accuracy"]) # tinh do chinh xac
# np.array([x[0] for _, x in enumerate(xtrain)])
# print(np.array([x[0] for _, x in enumerate(xtrain)]))
# print( np.array([y[1] for _, y in enumerate(xtrain)]))
# model_training.fit(np.array( [x[0] for _, x in enumerate(xtrain)] ), np.array( [y[1] for _, y in enumerate(xtrain)] ), epochs=10)
# # data train/label/loop
# #
# model_training.save('model' )
models = models.load_model('model')
listresult=['cuong','duc','vinh','dan']
import cv2

detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
count = 0
#
while True:
    OK, frame = cam.read()
    faces = detect_face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:
        img_face = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2],(100,100))
        result2 = model_training.predict(img_face.reshape(-1,100,100,3), verbose=0)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
        # cv2.imwrite('Picture/face_cam_{}.jpg'.format(count), img_face )
        # rate = np.argmax[result2]
        # if np.argmax(result2) > 0.9:
        #     cv2.putText(frame, listresult[result2], (x+15,y-15), cv2.FONT_HERSHEY_PLAIN,
        #             0.8, (255,25,255),2 )
        # cv2.putText(frame, str(rate), (x + 15, y - 30), cv2.FONT_HERSHEY_PLAIN,
        #             0.8, (255, 25, 255), 2)
        print(result2)
        cv2.putText(frame, listresult[np.argmax(result2)], (x + 15, y - 15), cv2.FONT_HERSHEY_PLAIN,
                                 0.8, (255,25,255),2 )
        count = count + 1
    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'): # print x to exit
        break

cam.release()
cv2.destroyAllWindows()