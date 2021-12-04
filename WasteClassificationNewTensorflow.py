import tensorflow as tf
import keras
from tensorflow.keras import layers, datasets, models, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import openpyxl
import time
import pathlib
import matplotlib.pyplot as plt


start = time.time()
data_dir = "C:/Users/Diego Flores/Pictures/ENSC 424 Waste Classification separated/Training"

# Directory of all models
#modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model/"
modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model (best) Custom/"
#modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model (best) AlexNet/"

# Where we save the model being trained
modelpathsave = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model/"

excelpath = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/ensc424_waste_classification_testcases.xlsx"
excelpage = "Custom"  # Custom or AlexNet

batch_size = 32
img_height = 227
img_width = img_height
epochs = 100
validationsplit = 0.2
dataseed = 272

data_dir = pathlib.Path(data_dir)
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=validationsplit,
    subset="training",
    seed=dataseed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=validationsplit,
    subset="validation",
    seed=dataseed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

##

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(
            img_height, img_width, 3)),
        layers.RandomRotation(0.5),
        layers.RandomZoom(-0.5),
    ]
)
trainChoice = 0
global model
while ((trainChoice != 'n') and (trainChoice != 'N') and (trainChoice != 'y') and (trainChoice != 'Y')):
    trainChoice = input("Resume training on previous model (y/n): ")
    if ((trainChoice == 'n') or (trainChoice == 'N')):

        # Flower model
        if excelpage == 'Custom':
            model = Sequential(
                [
                    data_augmentation,
                    layers.Rescaling(1.0 / 255),

                    # The first convolution
                    layers.Conv2D(16, 5, strides=2, activation='relu'),
                    layers.MaxPooling2D(),
                    # The second convolution
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    # The third convolution
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    # The Fourth convolution
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),

                    layers.Dropout(0.2),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.05),
                    layers.Dense(64, activation='relu'),

                    layers.Dense(num_classes)
                ]
            )

        # AlexNet model
        elif excelpage == 'AlexNet':
            model = models.Sequential()
            model.add(data_augmentation)
            layers.Rescaling(1.0 / 255),

            # The first convolution
            model.add(layers.Conv2D(96, 11, strides=4))
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling2D(3, strides=2))

            # The second convolution
            model.add(layers.Conv2D(256, 5, padding='same'))
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling2D(3, strides=2))

            # The third convolution
            model.add(layers.Conv2D(384, 3, padding='same'))
            model.add(layers.Activation('relu'))

            # The fourth convolution
            model.add(layers.Conv2D(384, 3, padding='same'))
            model.add(layers.Activation('relu'))

            # The fifth convolution
            model.add(layers.Conv2D(256, 3, padding='same'))
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling2D(3, strides=2))

            # fully connected layer 1
            model.add(layers.Flatten())
            model.add(layers.Dense(4096, activation='relu'))
            model.add(layers.Dropout(0.5))
            # fully connected layer 2
            model.add(layers.Dense(4096, activation='relu'))
            model.add(layers.Dropout(0.5))
            # fully connected layer softmax
            model.add(layers.Dense(num_classes, activation='softmax'))

    elif((trainChoice == 'y') or (trainChoice == 'Y')):
        model = models.load_model(modelpathload, compile=False)

    else:
        print("Did not enter a valid option (y/n)")
        print("Please try again")


model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

callback = tf.keras.callbacks.EarlyStopping(  # stop the run if the loss starts increasing
    monitor='val_loss',
    patience=15,
    mode='min',
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[callback]
)


model.save(modelpathsave)

end = time.time()   # end timer
dur = end-start     # calculate duration


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

# Plotting data using matlab commands
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

# Code for putting info in Excel Page
wb = openpyxl.load_workbook(excelpath)
sheet = wb.get_sheet_by_name(excelpage)

exIndex = 1
while sheet['A' + str(exIndex + 2)].value != None:
    exIndex += 1

largest_val_acc = max(val_acc)
optIndex = val_acc.index(largest_val_acc)
sheet['A' + str(exIndex+2)].value = exIndex
# B - BATCH number
sheet['B' + str(exIndex+2)].value = batch_size
# C - Validation Split
sheet['C' + str(exIndex+2)].value = validationsplit
# D - Dataset Seed
sheet['D' + str(exIndex+2)].value = dataseed
# E - Image Size
sheet['E' + str(exIndex+2)].value = img_height
# F - Standarization
sheet['F' + str(exIndex+2)].value = "yes"
# G - Num Epochs (Best Performance)
sheet['G' + str(exIndex+2)].value = optIndex
# H - Training Accuracy
sheet['H' + str(exIndex+2)].value = acc[optIndex]
# I - Validation Accuracy
sheet['I' + str(exIndex+2)].value = largest_val_acc
# J - Training Loss
sheet['J' + str(exIndex+2)].value = loss[optIndex]
# K - Validation Loss
sheet['K' + str(exIndex+2)].value = val_loss[optIndex]
# L - TIME
sheet['L' + str(exIndex+2)].value = dur/60


print("")
if dur < 60:
    print("Execution Time:", dur, "seconds")
elif dur > 60 and dur < 3600:
    dur = dur/60
    print("Execution Time:", dur, "minutes")
else:
    dur = dur/(60*60)
    print("Execution Time:", dur, "hours")

wb.save(excelpath)
plt.show()
