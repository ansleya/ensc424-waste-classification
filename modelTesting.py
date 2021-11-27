import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange
from PIL import Image
import random
import time
random.seed = time.time()


#modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model/"
modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model (best) flower/"
# modelpathload = "C:/Users/Diego Flores/OneDrive - sfuca0/SFU/Fall 2021/ENSC 424/Project/model (best) AlexNet/"

test_base = "C:/Users/Diego Flores/Pictures/ENSC 424 Waste Classification separated/Testing/"

# PARAMETERS | PREFERENCES
batch_size = 32
img_height = 227
img_width = img_height
dataseed = 272
numPicturesDisplayed = 4  # number of pictures displayed per category

# INITIALIZATIONS
imagearray = []
imagePrediction = []
total = 0
totalcorrect = 0
pictureflag = 0
randImgCount = 0
randImgFlg = 0


test_ds = image_dataset_from_directory(
    test_base,
    seed=dataseed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = test_ds.class_names
num_classes = len(class_names)


model = models.load_model(modelpathload, compile=False)

for i in range(0, num_classes):
    input_dir = test_base + class_names[i]
    files = os.listdir(input_dir)
    name = class_names[i]

    # to keep pictures unique in demonstration
    # max(randImgFlg)*numPicturesDisplayed images in category to guarantee numPicturesDisplayed pitures
    randImgFlg = random.randrange(1, int(len(files)/numPicturesDisplayed), 1)

    print(f'     Now Testing {name} Images with Model...')

    for j in trange(0, len(files)):

        random_file = files[j]
        img = tf.keras.utils.load_img(
            input_dir + "/" + random_file, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if randImgCount == randImgFlg and pictureflag < numPicturesDisplayed:

            imagearray.append(Image.open(input_dir + "/" + random_file))
            imagePrediction.append(class_names[np.argmax(score)])

            randImgFlg = random.randrange(1, 15, 1)
            randImgCount = 0
            pictureflag += 1

        if name == class_names[np.argmax(score)]:
            totalcorrect += 1

        randImgCount += 1
        total += 1

    randImgCount = 0
    pictureflag = 0
    print('')

testAccuracy = totalcorrect/total * 100

print(
    f'Testing model with test images yielded a {testAccuracy:.2f} % accuracy ')
print("")


fig = plt.figure(figsize=(numPicturesDisplayed+1, num_classes+1))
# grid for pairs of subplots
grid = plt.GridSpec(num_classes, 1)

for i in range(num_classes):
    # create fake subplot just to title set of subplots
    fake = fig.add_subplot(grid[i])
    # '\n' is important
    fake.set_title(f'{class_names[i]}:\n', fontweight='semibold', size=14)
    fake.set_axis_off()

    gs = gridspec.GridSpecFromSubplotSpec(
        1, numPicturesDisplayed, subplot_spec=grid[i])

    for j in range(numPicturesDisplayed):

        # real subplot
        ax = fig.add_subplot(gs[j])
        ax.set_title(
            imagePrediction[i*numPicturesDisplayed + j % numPicturesDisplayed])

        ax.imshow(imagearray[i*numPicturesDisplayed + j %
                  numPicturesDisplayed], aspect='auto')

        ax.axis("off")


fig.patch.set_facecolor('white')
fig.suptitle('Test Images with Predictions', fontweight='bold', size=16)
fig.tight_layout()
plt.show()
