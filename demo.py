'''########### Making single predictions ###########'''
import pickle
import numpy as np
from keras.preprocessing import image
from tensorflow.keras import models

classifier = models.load_model('model')
with open('ResultsMap.pkl', 'rb') as f:
  ResultMap = pickle.load(f)

ImagePath='./dataset/Face Images/Final_Training_Images/cr7/Screenshot 2022-05-03 132119.png'
# ImagePath='./dataset/Face Images/Final Training Images/train Cuong/face_cam_1.jpg'
# ImagePath='./dataset/Face Images/Final Training Images/train_duc/face_cam_2.jpg'
ImagePath='./dataset/Face Images/Final Training Images/Hoan/IMG_1290.JPG'
# ImagePath='./dataset/Face_Images/Final_Training_Images/Vinh/face_cam_171.jpg'
# ImagePath='./dataset/Face Images/Final Testing Images/face8/1face8.jpg'
# ImagePath='./dataset/Face Images/Final Testing Images/Einstein/Einstein.png'
ImagePath='./dataset/Face Images/Final Testing Images/Einstein/Screenshot 2022-05-03 135409.png'

test_image=image.load_img(ImagePath,target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image.reshape((-1,64,64,3)),verbose=0)
print(result)


if np.max(result) > 0.9:
  print(np.max(result))
  print('Prediction is: ',ResultMap[np.argmax(result)])
else:
  print(np.max(result))
  print("???????")
import os, cv2

detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

def cam():
    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        OK, frame = cam.read()
        faces = detect_face.detectMultiScale(frame, 1.3, 5)

        for (x,y,w,h) in faces:
            img_face = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2],(64,64))
            img_face = image.img_to_array(img_face)
            img_face = np.expand_dims(img_face, axis=0)
            result2 = classifier.predict(img_face.reshape(-1,64,64,3), verbose=0)
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
            # cv2.imwrite('Picture/face_cam_{}.jpg'.format(count), img_face )
            # print(np.max(result2))
            if np.max(result2) > 0.5:
                cv2.putText(frame, str(ResultMap[np.argmax(result2)]), (x+15,y-15), cv2.FONT_HERSHEY_PLAIN,
                        0.8, (255,25,255),2 )
            # cv2.putText(frame, str(rate), (x + 15, y - 30), cv2.FONT_HERSHEY_PLAIN,
            #             0.8, (255, 25, 255), 2)

            count = count + 1
        cv2.imshow('face', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'): # print x to exit
            break

    cam.release()
    cv2.destroyAllWindows()
#demo input img
img_path = './dataset/Face Images/Final Training Images/Hoan/IMG_1290.JPG'
img_path='./dataset/Face Images/Final Training Images/cr7/Screenshot 2022-05-03 132119.png'
# img_path='./dataset/Face Images/Final Training Images/train Cuong/face_cam_1.jpg'
# img_path='./dataset/Face Images/Final Training Images/train_duc/face_cam_76.jpg'
# img_path='./dataset/Face Images/Final Training Images/Hoan/IMG_1290.JPG'
# img_path='./dataset/Face Images/Final Training Images/Vinh/face_cam_179.jpg'
# img_path='./dataset/Face Images/Final Testing Images/face11/1face11.jpg'
# img_path='./dataset/Face Images/Final Testing Images/Einstein/Einstein.png'
# img_path='./dataset/Face Images/Final Testing Images/Einstein/Screenshot 2022-05-03 135409.png'

def detect(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(500,500))
    faces = detect_face.detectMultiScale(img, 1.1, 3 )
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    count = 0
    for (x,y,w,h) in faces:
        img_face= cv2.resize(img[y:y+h, x:x+w], (64,64))
        result2 = classifier.predict(test_image.reshape(-1, 64, 64, 3), verbose=0)
        cv2.rectangle( img , (x,y), (x+w,y+h), (70,170,120), 1 )
        if np.max(result2) > 0.9:
            cv2.putText(img,str(ResultMap[np.argmax(result2)]), (x+15,y-15), cv2.FONT_HERSHEY_PLAIN,
                     0.8, (255,25,255),2)
        else:
            cv2.putText(img, "unknow", (x + 15, y - 15), cv2.FONT_HERSHEY_PLAIN,
                        0.8, (255, 25, 255), 2)
        print(result2)
        print(np.max(result2))
        count = count +1
    while cv2.waitKey(1) & 0xFF != ord('q'):
        cv2.imshow('face', img)
        # milisecond;  0xFF = exit
# detect(img_path)
cam()
