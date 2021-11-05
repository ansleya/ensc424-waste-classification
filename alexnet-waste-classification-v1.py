import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models, losses
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

from pathlib import Path
import os
import zipfile
# https://www.tensorflow.org/tutorials/images/classification
## Loading the data as a data object

train_ds = tf.keras.utils.image_dataset_from_directory(directory = "C:\\Users\\ansle\\OneDrive\\Documents\\SFU\\Fourth Year\\Fall 2021\\ENSC 424\\training classes\\",
                                                      validation_split=0.2,
                                                      seed = 123,
                                                      subset="training",
                                                      labels = "inferred",
                                                      image_size = (227,227))

val_ds = tf.keras.utils.image_dataset_from_directory(directory = "C:\\Users\\ansle\\OneDrive\\Documents\\SFU\\Fourth Year\\Fall 2021\\ENSC 424\\training classes\\",
                                                      validation_split=0.2,
                                                      seed = 123,
                                                      subset="validation",
                                                      labels = "inferred",
                                                      image_size = (227,227))

##

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential()
model.add(layers.Rescaling(1./255, input_shape=(227, 227, 3)))
# The first convolution
model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
# The second convolution
model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
# The third convolution
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
# The fourth convolution
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
# The fifth convolution
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
# fully connected layer 1
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
# fully connected layer 2
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
# fully connected layer softmax
model.add(layers.Dense(8, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
#history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("wastec.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
