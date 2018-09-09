from data.model import create_model
from img_dataloader import TripletGenerator
from data.triplet import TripletLossLayer
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os

WEIGHTS_PATH = './weights/nn4.small2.final.hdf5'
WEIGHTS_CALLBACK = os.path.join('./weights', 'nn4.small2.{epoch:02d}.hdf5')

model = create_model()

in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_output')([emb_a, emb_p, emb_n])

train = Model([in_a, in_p, in_n], triplet_loss_layer)

generator = TripletGenerator()
dataset_a, dataset_p, dataset_n, _ = generator.generator_triplet_loss(1600, './out_img')

train.compile(loss=None, optimizer='adam')

save_weights = ModelCheckpoint(WEIGHTS_CALLBACK, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=25)

train.fit([dataset_a, dataset_p, dataset_n], None, shuffle=True, batch_size=4, epochs=300, callbacks=[save_weights])

train.save_weights(WEIGHTS_PATH)

