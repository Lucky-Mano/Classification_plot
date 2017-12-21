# -*- coding: utf-8 -*-
import matplotlib as mpl
import argparse
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from auto_encoder.AE import AutoEncoder, StackedAutoEncoder

mpl.use('QT5Agg')
plt.rcParams['figure.figsize'] = [12, 9]

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', '-n',
                        default='500,250,100,2',
                        help='num neurons of layers')
    parser.add_argument('--epochs', '-e', type=int,
                        default=500,
                        help='epochs')
    parser.add_argument('--batch', '-b', type=int,
                        default=512,
                        help='batch size')

    args = parser.parse_args().__dict__
    neurons = list(map(int, args.pop('neurons').split(',')))
    epochs = args.pop('epochs')
    batch_size = args.pop('batch')

    (x_train, x_test), (y_train, y_test) = mnist.load_data()
    data = np.concatenate((x_train, y_train)) / 255.0
    label = np.concatenate((x_test, y_test))

    num_input = data.shape[1] * data.shape[2]
    num_neurons = [num_input]
    num_neurons.extend(neurons)

    data = data.reshape(-1, num_input)

    aes = []
    for i in range(len(num_neurons) - 1):
        if i == 0:
            aes.append(AutoEncoder(num_neurons[i], num_neurons[i + 1],
                                   out_activation="linear"))

        elif i == len(num_neurons) - 1:
            aes.append(AutoEncoder(num_neurons[i], num_neurons[i + 1],
                                   mid_activation="linear"))

        else:
            aes.append(AutoEncoder(num_neurons[i], num_neurons[i + 1]))

    train_data = None
    for ae in aes:
        if train_data is None:
            train_data = data

        ae.compile()
        ae.train(train=train_data, epochs=epochs, batch_size=batch_size)
        train_data = ae.encoder.predict(train_data)

    sae = StackedAutoEncoder(num_neurons)
    sae.load_weight(aes)
    sae.compile()
    sae.train(data, epochs, batch_size)
    reduced = sae.encoder.predict(data)

    for i in range(10):
        num = reduced[label == i]
        plt.scatter(num[:, 0], num[:, 1], s=5, label=f'{i}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
