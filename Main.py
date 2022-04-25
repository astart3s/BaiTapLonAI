import cv2
import os
image_path_parent = 'Picture'
detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')


# image_path = 'Picture/pic1.jfif'

def detect(img_path):
    img_path  = cv2.imread(img_path)
    faces = detect_face.detectMultiScale(img_path, 1.1, 3 )
     # tra ve doi tuong 4 tham so
     # (img, scalefactor:ti le giam diem anh, minNeighbor: )
    count = 0
    for (x,y,w,h) in faces:
        img_face= cv2.resize(img_path[y:y+h, x:x+w], (100,100))
        cv2.imwrite('Picture/train/face_{}.png'.format(count), img_face)

        cv2.rectangle( img_path, (x,y), (x+w,y+h), (70,170,120), 1 )
        count = count +1

    while cv2.waitKey(1)& 0xFF != ord('q'):
        cv2.imshow('face', img_path)
        # milisecond;  0xFF = exit

for i in os.listdir(image_path_parent):
    img_path = os.path.join((image_path_parent), i )
    print(img_path)
    if img_path.endswith('jpg'):
        detect(img_path)


