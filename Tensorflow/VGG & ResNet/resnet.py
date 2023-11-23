import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from glob import glob
import os


train_path = "./fruits_360/Training"
test_path = "./fruits_360/Validation"
batch_size = 32

def ShowImage():
    img = load_img(train_path + "/Lemon/0_100.jpg")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def LoadModel(numberOfClass):
    #import inception with pre-trained weights. do not include fully #connected layers
    inception_base = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = inception_base.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    # and a fully connected output/classification layer
    predictions = Dense(numberOfClass, activation='softmax')(x)
    # create the full network so we can train on it
    model = Model(inputs=inception_base.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
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


def Generator():
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path, target_size=(224, 224), batch_size=batch_size, class_mode='categorical'
    )

    return train_generator, test_generator



if __name__ == '__main__':
    os.system('clear')

    ShowImage()

    numberOfClass = len(glob(train_path + "/*"))
    print("Number of classes: ", numberOfClass)

    train_gen, test_gen = Generator()
    os.system('clear')

    #load model
    model = LoadModel(numberOfClass)
    print(model.summary())

    #train model with gpu
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    print(gpu_device)
    with tf.device("/device:GPU:0"):
        r = model.fit(train_gen, epochs=5, shuffle = True, verbose = 1, validation_data = test_gen)
    
    #result
    Visualize(r)
