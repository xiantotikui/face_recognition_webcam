from data.model import create_model
from img_dataloader import TripletGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from keras.layers import Subtract, Concatenate
from keras import backend as K
import os
import pickle
import numpy as np

READY_WEIGHTS_PATH = './weights/nn4.small2.final.hdf5'

WEIGHTS_PATH = './weights/transfer.final.hdf5'
WEIGHTS_CALLBACK = os.path.join('./weights', 'transfer.{epoch:02d}.hdf5')

NUMBER_CLASSES = 31

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

train = Model([in_a, in_p, in_n], [emb_a, emb_p, emb_n])
train.load_weights(READY_WEIGHTS_PATH)

for layer in train.layers:
    layer.trainable = False

a, p, n = train.output
dist = Subtract()([p, n])
x = Concatenate()([a, dist])
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(NUMBER_CLASSES, activation="softmax")(x)

transfer = Model(train.input, predictions)

transfer.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

generator = TripletGenerator()
dataset_a, dataset_p, dataset_n, data_y = generator.generator_triplet_loss(NUMBER_CLASSES, './out_img')

save_weights = ModelCheckpoint(WEIGHTS_CALLBACK, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)

with open(os.path.join('./labels', 'labels_100.pb'), 'rb') as fp:
    labels_file = pickle.load(fp)

labels = []
for y in data_y:
	for key, val in labels_file.items():
		if y == key:
			labels.append(val)

transfer.fit([dataset_a, dataset_p, dataset_n], np.asarray(labels), shuffle=True, batch_size=4, epochs=120, callbacks=[save_weights])

transfer.save_weights(WEIGHTS_PATH)
