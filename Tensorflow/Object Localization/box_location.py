import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from matplotlib.patches import Rectangle

def create():
  vgg = tf.keras.applications.VGG16(input_shape=[100, 100, 3], include_top=False, weights='imagenet')
  x = Flatten()(vgg.output)
  x = Dense(4, activation='sigmoid')(x)
  model = Model(vgg.input, x)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model

def image_generator(batch_size=64):
  # generate image and targets
  while True:
    for _ in range(50):
      X = np.zeros((batch_size, 100, 100, 3))
      Y = np.zeros((batch_size, 4))
      
      for i in range(batch_size):
        # make the boxes and store their location in target
        row0 = np.random.randint(90)
        col0 = np.random.randint(90)
        row1 = np.random.randint(row0, 100)
        col1 = np.random.randint(col0, 100)
        X[i,row0:row1,col0:col1,:] = 1
        Y[i,0] = row0/100.
        Y[i,1] = col0/100.
        Y[i,2] = (row1 - row0)/100.
        Y[i,3] = (col1 - col0)/100.
      
      yield X, Y

def make_prediction():
  # Generate a random image
  x = np.zeros((100, 100, 3))
  row0 = np.random.randint(90)
  col0 = np.random.randint(90)
  row1 = np.random.randint(row0, 100)
  col1 = np.random.randint(col0, 100)
  x[row0:row1,col0:col1,:] = 1
  print(row0, col0, row1, col1)
  
  # Predict
  X = np.expand_dims(x, 0)
  p = model.predict(X)[0]
    
  # Draw the box
  fig, ax = plt.subplots(1)
  ax.imshow(x)
  rect = Rectangle((p[1]*100, p[0]*100), p[3]*100, p[2]*100,linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()

if __name__ == '__main__':
  #create model
  model = create()
  model.fit(image_generator(), steps_per_epoch=50, epochs=5)
  while True:
    make_prediction()
    val = str(input('Continue to predict?(Y/N): '))
    if val.lower() != 'y': break