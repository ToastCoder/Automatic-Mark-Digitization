#-------------------------------------------------------------------------------------------------------------------------------

# AUTOMATIC MARK DIGITIZATION

# GETS THE IMAGE FROM THE USER AND EXTRACTS THE RESPECTIVE ROLL.NO AND THEIR MARK AND UPDATES IT IN A .CSV FILE.

# FILE NAME: MAIN.PY

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Multiclass Classification

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# DATA SEGMENTATION
data = pd.read_csv('data/train.csv')