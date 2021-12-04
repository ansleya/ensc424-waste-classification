import random
import shutil
import math
import os
import time
from scipy.sparse import base


categories = [
    "batteries",
    "cardboard",
    "fabric",
    "glass",
    "landfill",
    "metal",
    "organic",
    "paper",
    "plastic"
]
main_dir_base = (
    "C:/Users/Diego Flores/Pictures/ENSC 424 Waste Classification separated/Training/"
)
val_base = (
    "C:/Users/Diego Flores/Pictures/ENSC 424 Waste Classification separated/Validation/"
)
test_base = (
    "C:/Users/Diego Flores/Pictures/ENSC 424 Waste Classification separated/Testing/"
)

# method = partition: to split folders with specific sizings
# method = merge: to cut files from input folder and paste in ouput folder

method = "partition"

validation_split = 0
testing_split = 0.1

random.seed = time.time()

if method == "partition":
    for i in range(0, len(categories)):
        input_dir = main_dir_base + categories[i]
        output_val = val_base + categories[i]
        output_test = test_base + categories[i]
        #######################
        split = validation_split + testing_split
        files = os.listdir(input_dir)
        num_val = math.floor(len(files) * validation_split)
        num_test = math.floor(len(files) * testing_split)

        for i in range(num_val):
            random_file = random.choice(files)
            shutil.move(input_dir + "/" + random_file, output_val)
            files.remove(random_file)

        for i in range(num_test):
            random_file = random.choice(files)
            shutil.move(input_dir + "/" + random_file, output_test)
            files.remove(random_file)


elif method == "merge":

    for i in range(0, len(categories)):
        input_dir = (
            test_base + categories[i]
        )  # Folder where you one files to be cut (val_base or test_base)
        output_dir = (
            main_dir_base + categories[i]
        )  # Folder where you one files to be pasted (main_dir_base)

        #######################
        files = os.listdir(input_dir)
        num_val = math.floor(len(files))

        for i in range(num_val):
            random_file = random.choice(files)
            shutil.move(input_dir + "/" + random_file, output_dir)
            files.remove(random_file)


else:
    print("not a proper action")
