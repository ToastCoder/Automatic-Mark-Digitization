#-------------------------------------------------------------------------------------------------------------------------------

# AUTOMATIC MARK DIGITIZATION

# GETS THE IMAGE FROM THE USER AND EXTRACTS THE RESPECTIVE ROLL.NO AND THEIR MARK AND UPDATES IT IN A .CSV FILE.

# FILE NAME: TEST.PY

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Multiclass Classification

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED LIBRARIES
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# FUNCTION FOR IMAGE PREPROCESSING
def image_preprocessing(img):
    return None

samples_dir = './samples'
im = []

for i in os.listdir(samples_dir):
    i = os.path.join(samples_dir,im)
    im.append(Image.open(i))
    
files = os.listdir(samples_dir)
