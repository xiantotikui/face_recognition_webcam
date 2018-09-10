import cv2
import sys
import numpy as np
import os

IMG_SIZE = (96, 96)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

i = 0
while i < 120:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE)
        
    if i % 8 == 0:
        cv2.imwrite(os.path.join('./tmp', str(i) + '.jpg'), img)
        
    i += 1
    
video_capture.release()
cv2.destroyAllWindows()
