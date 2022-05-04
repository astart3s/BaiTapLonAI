import os, cv2

# from celery.bin.result import result

detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

TrainingImagePath= 'D:\ky6\\ai\\btl\BaiTapLonAI-master\Picture\Face_Images\Final_Training_Images'

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)


test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Printing class labels for each face
test_set.class_indices
print(test_set)
TrainClasses=training_set.class_indices

ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

import pickle
# with open("ResultsMap.pkl", 'wb') as fileWriteStream:
#     pickle.dump(ResultMap, fileWriteStream)

print("Mapping of Face and its ID",ResultMap)
# print(ResultMap[1][0]+"fsadf")
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)


from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras import models
import tensorflow as tf

# create model
classifier= models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), activation='relu'),

    # '''# STEP--2 MAX Pooling'''
    layers.MaxPool2D((2,2)),

    layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    layers.MaxPool2D((2, 2)),
    # layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    # layers.MaxPool2D((2, 2)),

    layers.Flatten(),

    layers.Dense(1000, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(OutputNeurons, activation='softmax')
])
# model_training = models.Sequential([

classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])


classifier.fit(training_set, epochs=16, validation_data=test_set)
classifier.save('model_complete')
# import matplotlib
# test_model = models.load_model('model_complete')



import numpy as np
from keras.preprocessing import image

ImagePath= './Picture/Face Images/Final Training Images'
ImagePath='./dataset/Face Images/Final Testing Images/face5/3face5.jpg'
test_image=image.load_img(ImagePath,target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image.reshape((-1,64,64,3)),verbose=0)
print(result)
print(training_set.class_indices)

print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])

import os, cv2
detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
count = 0
#
# while True:
#     OK, frame = cam.read()
#     faces = detect_face.detectMultiScale(frame, 1.3, 5)
#
#     for (x,y,w,h) in faces:
#         img_face = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2],(64,64))
#         result2 = classifier.predict(img_face.reshape(-1,64,64,3), verbose=0)
#         cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
#         # cv2.imwrite('Picture/face_cam_{}.jpg'.format(count), img_face )
#         # rate = np.argmax[result2]
#         if np.argmax(result2) > 0.9:
#             cv2.putText(frame, str(ResultMap[np.argmax(result2)]), (x+15,y-15), cv2.FONT_HERSHEY_PLAIN,
#                     0.8, (255,25,255),2 )
#         # cv2.putText(frame, str(rate), (x + 15, y - 30), cv2.FONT_HERSHEY_PLAIN,
#         #             0.8, (255, 25, 255), 2)
#
#         count = count + 1
#     cv2.imshow('face', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('x'): # print x to exit
#         break
#
# cam.release()
# cv2.destroyAllWindows()

# img_path = 'H:/unique TRẦN NHÂN TÔNG_/100CANON/LÂM/IMG_7992.jpg'

img_path = 'C:/Users/T460s/Downloads/da8116e1fefc3fa266ed.jpg'
def detect(img_path):
    img_path  = cv2.imread(img_path)
    img_path= cv2.resize(img_path,(500,500))
    faces = detect_face.detectMultiScale(img_path, 1.1, 3 )
     # tra ve doi tuong 4 tham so
     # (img, scalefactor:ti le giam diem anh, minNeighbor: )
    count = 0
    for (x,y,w,h) in faces:
        img_face= cv2.resize(img_path[y:y+h, x:x+w], (64,64))
        # cv2.imwrite('Picture/train/face_{}.png'.format(count), img_face)
        result2 = test_model.predict(img_face.reshape(-1, 64, 64, 3), verbose=0)
        cv2.rectangle( img_path, (x,y), (x+w,y+h), (70,170,120), 1 )
        cv2.putText(img_path,str(ResultMap[np.argmax(result2)]), (x+15,y-15), cv2.FONT_HERSHEY_PLAIN,
                     0.8, (255,25,255),2)
        print(result2)
        count = count +1

    while cv2.waitKey(1) & 0xFF != ord('q'):
        cv2.imshow('face', img_path)
        # milisecond;  0xFF = exit
# detect(img_path)


# for i in os.listdir(image_path_parent):
#     img_path = os.path.join((image_path_parent), i )
#     print(img_path)
#     if img_path.endswith('jpg'):
#         detect(img_path)
