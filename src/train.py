#-------------------------------------------------------------------------------------------------------------------------------

# AUTOMATIC MARK DIGITIZATION

# GETS THE IMAGE FROM THE USER AND EXTRACTS THE RESPECTIVE ROLL.NO AND THEIR MARK AND UPDATES IT IN A .CSV FILE.

# FILE NAME: MAIN.PY

# DONE BY: VIGNESHWAR RAVICHANDAR

# TOPICS: Deep Learning, TensorFlow, Convolutional Neural Networks, Multiclass Classification

#-------------------------------------------------------------------------------------------------------------------------------

# IMPORTING REQUIRED LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('ggplot')

print(f"TensorFlow Version: {tf.__version__}")

TRAIN_DATASET_DIR = 'data/train.csv'
TEST_DATASET_DIR = 'data/test.csv'
MODEL_PATH = 'model/digit_model'

# DATA PREPROCESSING
data_train = pd.read_csv(TRAIN_DATASET_DIR)

print("Dataset Description:\n",data_train.describe())
print("Dataset Head:\n",data_train.head())

y_train = data_train['label']
x_train = data_train.drop('label',axis = 1)
x_test = pd.read_csv(TEST_DATASET_DIR)

# NORMALIZATION OF X AXIS
x_train_norm = tf.keras.utils.normalize(x_train, axis = 1)
x_test_norm = tf.keras.utils.normalize(x_test, axis = 1)

# FUNCTION FOR NEURAL NETWORK
def digit_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512,activation = 'relu'))
    model.add(tf.keras.layers.Dense(128,activation = 'relu'))
    model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
    return model

# INITITIALIZING THE CALLBACK
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', mode = 'max')

# FITTING AND TRAINING THE MODEL
model = digit_model()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train_norm, y_train, validation_split = 0.1, epochs = 30,callbacks = early_stopping, batch_size = 5)
model.summary()

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/loss_graph.png')

# PLOTTING THE GRAPH FOR TRAIN-ACCURACY AND VALIDATION-ACCURACY
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
plt.savefig('graphs/acc_graph.png')


