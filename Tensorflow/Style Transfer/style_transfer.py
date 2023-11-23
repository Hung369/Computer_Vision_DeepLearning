import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def show_image(image):
    plt.imshow(np.squeeze(image))
    plt.show()

if __name__=="__main__":
    os.system('clear')
    
    # Load model
    model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    os.system('clear')

    content_img = load_image("loz.jpg")
    style_img = load_image("Mona_Lisa.jpg")

    show_image(content_img)
    show_image(style_img)

    # style transfer
    stylized_img = model(tf.constant(content_img), tf.constant(style_img))[0]
    show_image(stylized_img)