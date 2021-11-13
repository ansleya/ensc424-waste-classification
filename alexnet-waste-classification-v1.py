from logging import NullHandler
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

import openpyxl
# https://www.tensorflow.org/tutorials/images/classification
loadDir = input("Type in directory of previous model (enter nothing to start from scratch): ")
loadDir = Path(loadDir)

## Loading the data as a data object
# set values for preparing the dataset below
inVal_Split = 0.2 # splits data into training set and validation set (eg. 0.2 val -> 20% of files are for validation)
inSeed = 123      # sets the seed so that the library remembers how they split up the above
inImgSize = 227 
saveDir = r'C:\Users\ansle\OneDrive\Documents\SFU\Fourth Year\Fall 2021\ENSC 424'
saveDir = Path(saveDir)
if (not saveDir.is_dir()):
  print("Error: invalid save directory")
  exit()
############### need to sort data into directories - the function creates classes based on the number of folders
train_ds = tf.keras.utils.image_dataset_from_directory(directory = r'C:\Users\ansle\OneDrive\Documents\SFU\Fourth Year\Fall 2021\ENSC 424\ansley-diego-dataset',
                                                      validation_split=inVal_Split,
                                                      seed = inSeed,
                                                      subset="training",
                                                      labels = "inferred",
                                                      batch_size = 128,
                                                      image_size = (inImgSize,inImgSize))

val_ds = tf.keras.utils.image_dataset_from_directory(directory = r'C:\Users\ansle\OneDrive\Documents\SFU\Fourth Year\Fall 2021\ENSC 424\ansley-diego-dataset',
                                                      validation_split=inVal_Split,
                                                      seed = inSeed,
                                                      subset="validation",
                                                      labels = "inferred",
                                                      batch_size = 128,
                                                      image_size = (inImgSize,inImgSize))

##### for speeding up the runtime

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
global model
if (not loadDir.is_dir()): 
  ####### the CNN
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
else:
  model = models.load_model(loadDir,
                            compile = False
  )


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 100
callback = tf.keras.callbacks.EarlyStopping( # stop the run if the loss starts increasing
  monitor='val_loss',
  patience = 3,
  mode = 'min',
)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks = [callback]
)

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
model.save(saveDir)

wb = openpyxl.load_workbook('ensc424_waste_classification_testcases.xlsx')
sheet = wb.get_sheet_by_name('AlexNet')

i = 3
while sheet['A' + str(i)].value != None:
  i += 1


largest_val_acc = max(val_acc)
optIndex  = val_acc.index(largest_val_acc)
sheet['B' + str(i)].value = "CPU"            #B - GPU or CPU
sheet['C' + str(i)].value = inVal_Split      #C - Validation Split
sheet['D' + str(i)].value = inSeed           #D - Dataset Seed
sheet['E' + str(i)].value = inImgSize        #E - Image Size
sheet['F' + str(i)].value = "yes"            #F - Standarization
sheet['G' + str(i)].value = optIndex + 1             #G - Num Epochs (Best Performance)
sheet['H' + str(i)].value = acc[optIndex]              #H - Training Accuracy
sheet['I' + str(i)].value = largest_val_acc          #I - Validation Accuracy
sheet['J' + str(i)].value = loss[optIndex]           #J - Training Loss
sheet['K' + str(i)].value = val_loss[optIndex]        #K - Validation Loss
sheet['L' + str(i)].value = None    #L - Model File Name




wb.save('ensc424_waste_classification_testcases.xlsx')

