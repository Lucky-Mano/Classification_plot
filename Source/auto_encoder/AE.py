# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, Dense, Reshape

class AutoEncoder(object):
    def __init__(self, num_input, num_neuron,
                 mid_activation='sigmoid', out_activation="sigmoid"):
        inp = Input(shape=(num_input, ))

        enced = Dense(units=num_neuron, activation=mid_activation)(inp)

        deced = Dense(num_input, activation=out_activation)(enced)
        
        self.encoder = Model(inputs=inp, outputs=enced)
        self.autoencoder = Model(inputs=inp, outputs=deced)

    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.autoencoder.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def train(self, train=None,
              epochs=1, batch_size=128):
        self.autoencoder.fit(
            train, train,
            epochs=epochs,
            batch_size=batch_size
        )

class StackedAutoEncoder(object):
    def __init__(self, num_neurons):
        length = len(num_neurons)
        inp = Input(shape=(num_neurons[0], ))
        enced = inp
        for i, num in enumerate(num_neurons[1:]):
            if i == length - 2:
                enced = Dense(num, activation='linear')(enced)
            else:
                enced = Dense(num, activation='sigmoid')(enced)

        deced = enced
        for i, num in enumerate(reversed(num_neurons[:-1])):
            if i == length - 2:
                deced = Dense(num, activation='linear')(deced)
            else:
                deced = Dense(num, activation='sigmoid')(deced)

        self.encoder = Model(inputs=inp, outputs=enced)
        self.autoencoder = Model(inputs=inp, outputs=deced)

    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.autoencoder.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def train(self, train=None, epochs=1, batch_size=128):
        self.autoencoder.fit(train, train, epochs=epochs, batch_size=batch_size)

    def load_weight(self, aes):
        length = len(aes)

        for i, ae in enumerate(aes):
            r = length * 2 - i

            self.autoencoder.layers[i + 1].set_weights(ae.autoencoder.layers[1].get_weights())
            self.autoencoder.layers[r].set_weights(ae.autoencoder.layers[2].get_weights())
