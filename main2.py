import os, cv2

detect_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

#cam = cv2.VideoCapture(0)
count = 0
cam =cv2.VideoCapture('H:/DCIM/Camera/bb09303cf515d8fcd375be79562dd1d5.mp4')
while True:
    OK, frame = cam.read()
    faces = detect_face.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in faces:
        img_face = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2],(100,100))
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
        cv2.imwrite('Picture/face_cam_{}.jpg'.format(count), img_face )
        count = count + 1
    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()