import os
import cv2
from keras.utils import to_categorical
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

INPUT_DIR = './in_img'
OUTPUT_DIR = './out_img'
IMG_SIZE = (96, 96)

dir_files = os.listdir(os.path.join(INPUT_DIR))
labels = {}

for filename0 in os.listdir(INPUT_DIR):
    size = len(os.listdir(os.path.join(INPUT_DIR, filename0)))
    if size == 0 or size == 1:
        continue
    os.makedirs(os.path.join(OUTPUT_DIR, filename0))
    for filename1 in os.listdir(os.path.join(INPUT_DIR, filename0)):
        print(os.path.join(INPUT_DIR, filename0, filename1))
        label = to_categorical(np.where(np.asarray(dir_files) == os.path.join(filename0)), len(dir_files))[0]
        labels[filename0] = np.asarray(label)
        img = cv2.imread(os.path.join(INPUT_DIR, filename0, filename1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            tmp = cv2.resize(img[y:y+h, x:x+w], IMG_SIZE)

        cv2.imwrite(os.path.join(OUTPUT_DIR, filename0, filename1), tmp)

with open(os.path.join(OUTPUT_DIR, 'labels.pb'), 'wb') as fp:
    pickle.dump(labels, fp)
