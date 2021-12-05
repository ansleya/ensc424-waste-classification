# ensc424-waste-classification

Here is the completed ENSC 424 image classification project.

Ansley | Diego | Ray

## ENSC 424 Waste Classification Project - Group 6

### Python Files 
#### WasteClassificationNewTensorflow.py
This is the main program of our project.Here is where the model gets trained, the plots are constructed, and the data is stored in an excel file. Code structure based on tensorflow's turorial on image classification.

References: https://www.tensorflow.org/tutorials/images/classification 

#### modelTesting.py
This is where we test an existeing model with the testing data, finding its testing accuracy while asl generating a figure with images and the predicitons of those images with our model.

(This Code is fully custom)

#### filePartition.py
This is a quick way to partition the data files into the coreesponding customizable ratios of Training, Validation, and Testing.

(This Code is fully custom)

### Dependancies

Tensorflow -- version 2.7 (comes with keras -- 2.7)
matplotlib
openpyxl
time
pathlib
numpy
PIL
random
tqdm
