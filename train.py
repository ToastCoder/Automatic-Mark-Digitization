  
# AUTOMATIC MARK DIGITIZATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Convolutional Neural Networks, TensorFlow

# DISABLE TENSORFLOW DEBUG INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt