import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def Normalize(val1, val2):
    return val1 / 255.0, val2 / 255.0


def LossPlot(r):
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.show()


def AccPlot(r):
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.show()


if __name__ == '__main__':
    os.system('clear')

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = Normalize(x_train, x_test)
    print("X_train shape: ", x_train.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    os.system('clear')

    r = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=11)

    LossPlot(r)
    AccPlot(r)

    print(model.evaluate(x_test, y_test))
