from keras.layers import Layer
from keras import backend as K

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

# -----------------------------------------------------------------------------------------
# 04.09.2018 Code taken from https://github.com/keras-team/keras/issues/9498
# -----------------------------------------------------------------------------------------

    def triplet_loss(self, inputs, dist='sqeuclidean', margin='maxplus'):
        anchor, positive, negative = inputs
        positive_distance = K.square(anchor - positive)
        negative_distance = K.square(anchor - negative)
        if dist == 'euclidean':
            positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
        elif dist == 'sqeuclidean':
            positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
            negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
        loss = positive_distance - negative_distance
        if margin == 'maxplus':
            loss = K.maximum(0.0, 1 + loss)
        elif margin == 'softplus':
            loss = K.log(1 + K.exp(loss))
        return K.mean(loss)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return inputs
