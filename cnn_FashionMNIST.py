import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tensorflow as tf
from keras.utils import plot_model
from sklearn.metrics import classification_report

def Modeling():
    i = tf.keras.layers.Input(shape=x_train[0].shape)

    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(i)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(128,(3,3), activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(K, activation='softmax')(x)

    model = tf.keras.models.Model(i,x)
    return model



def Plotting(r, string):
    if string == 'loss':
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.show()
    if string == 'accuracy':
        plt.plot(r.history['accuracy'], label='accuracy')
        plt.plot(r.history['val_accuracy'], label='val_accuracy')
        plt.show()
    else:
        print('None')

def Prediction(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__=='__main__':
    os.system('clear')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train/255.0 , x_test/255.0
    print(x_train.shape)

    # adding new axis at the end
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print(x_train.shape)

    # extract labels
    K = len(set(y_train))
    print('numbers of labels = ', K)

    # Building CNN models
    model = Modeling()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # fit data
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
    Plotting(r, string='loss')
    Plotting(r, string='accuracy')

    os.system('clear')
    print(model.summary())   
    Prediction(model, x_test, y_test)
    