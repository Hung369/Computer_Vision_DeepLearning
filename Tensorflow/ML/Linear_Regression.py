import os
from io import StringIO
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def ReadCSV(url):
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data, header=None)
    print(df.info())
    return df.to_numpy()


def Scatter(X, Y):
    plt.scatter(X, Y)
    plt.show()


def schedule(epoch, lr):  # code by experience and intuition
    if epoch >= 50:
        return 0.0001
    return 0.001


if __name__ == "__main__":
    # Clearing the Screen
    os.system('clear')

    url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
    np_arr = ReadCSV(url)
    print(np_arr)

    X = np_arr[:, 0].reshape(-1, 1)  # making NxD matrix
    Y = np_arr[:, 1]
    Scatter(X, Y)

    Y = np.log(Y)
    Scatter(X, Y)
    X = X - X.mean()

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.001, 0.9),
        loss='mse',
    )

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    os.system('clear')
    r = model.fit(X, Y, epochs=200, callbacks=[scheduler])
    plt.plot(r.history['loss'], label='loss')
    plt.show()

    # get model's weights
    print(model.layers[0].get_weights)
    tf.keras.utils.plot_model(model, 'test.png')
