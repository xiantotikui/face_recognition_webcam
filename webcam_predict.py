import cv2
import sys
from collections import deque, Counter
from data.model import create_model
from img_dataloader import TripletGenerator
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Dense
from keras.layers import Subtract, Concatenate
import numpy as np
import os

WEIGHTS_TRANSFER_PATH = './weights/transfer.final.hdf5'

NUMBER_CLASSES = 31

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

transfer = Model([in_a, in_p, in_n], [emb_a, emb_p, emb_n])

for layer in transfer.layers:
    layer.trainable = False

x0, x1, x2 = transfer.output
x = Subtract()([x1, x2])
x = Concatenate()([x0, x])
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(NUMBER_CLASSES, activation="softmax")(x)

predict = Model(inputs=transfer.input, outputs=predictions)
predict.load_weights(WEIGHTS_TRANSFER_PATH)
predict.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

IMG_SIZE = (96, 96)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

frames = deque(maxlen=8)

times = []
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frames.append(cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE))

    if len(frames) >= 2:
        triplet = TripletGenerator()
        a, p, n = triplet.generator_from_webcam(4, './out_img', frames)
        predictions = predict.predict([a, p, n], batch_size=4, verbose=0)
        predictions = np.argmax(predictions, axis=1)
        times.append([predictions])
        cv2.imshow("Video", frames[-1])
    times_np_arr = np.asarray(times).flatten()
    print(Counter(times_np_arr).most_common(3))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
