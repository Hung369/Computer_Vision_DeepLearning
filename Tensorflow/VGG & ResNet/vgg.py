import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob
import os

train_path = "./fruits_360/Training"
test_path = "./fruits_360/Validation"

def ShowImage():
    img = load_img(train_path + "/Lemon/0_100.jpg")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def LoadModel(numberOfClass):
    vgg = VGG16()
    model = Sequential()

    # eliminate the last layer in VGG-16 modle
    for layer in vgg.layers[:-1]:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    model.add(Dense(numberOfClass, activation='softmax'))
    model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    return model


def Visualize(r):

    plt.plot(r.history["loss"], label = "training loss")
    plt.plot(r.history["val_loss"], label = "validation loss")
    plt.legend()
    plt.show()

    plt.plot(r.history["accuracy"], label = "training acc")
    plt.plot(r.history["val_accuracy"], label = "validation acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.system('clear')

    ShowImage()

    numberOfClass = len(glob(train_path + "/*"))
    print("Number of classes: ", numberOfClass)

    # loading model
    model = LoadModel(numberOfClass)
    print(model.summary())

    # data generator
    train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224))
    test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224))
    batch_size = 32

    # training model
    r = model.fit(train_data, steps_per_epoch = 1600 // batch_size, epochs = 25, 
                          validation_data = test_data, validation_steps = 800 // batch_size)
    
    # visualizing accuracy and loss
    Visualize(r)
