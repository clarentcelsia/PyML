# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:22:37 2022

@author: Allen
"""

# Libraries

from zipfile import ZipFile
from tensorflow.keras import optimizers
from tensorflow.keras import models
from shutil import unpack_archive
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

import os, glob # glob for image
import wget
import cv2
import pandas as pd
import numpy as np

# =============================================================================
# Jupyter Notebook for Downloading Zip File
# =============================================================================

#!{sys.executable} -m pip install wget
#!{sys.executable} -m pip install shutil
pd.set_option("display.max_columns", 101)

# Run this cell block to download and extract image data
# !wget 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/8c7bc0c59ee6bcdb3646a1997606a9d0/test.zip'
# !wget 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/ffe8804da24b94ee410a8730ba297dfc/train_0.zip'
# !wget 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/5460fcc87e2d7f7e19c17f719e0df00a/train_1.zip'

# unpack_archive('train_1.zip', '')
# unpack_archive('train_0.zip', '')
# unpack_archive('test.zip', 'test')

# print('Dataset Extracted')

# os.remove('train_1.zip')
# os.remove('train_0.zip')
# os.remove('test.zip')

# =============================================================================
# Download Zip Dataset File from URL 
# =============================================================================

test_zip = 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/8c7bc0c59ee6bcdb3646a1997606a9d0/test.zip'
train0_zip_ = 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/ffe8804da24b94ee410a8730ba297dfc/train_0.zip'
train1_zip_ = 'https://hr-projects-assets-prod.s3.amazonaws.com/1iaanii247i/5460fcc87e2d7f7e19c17f719e0df00a/train_1.zip'

def download_zip_from_url(zipdir, zipurl):
    for i in zipurl:
        wget.download(i, out=(zipdir))

# zips = [test_zip, train0_zip_, train1_zip_]
# download_zip_from_url('C:\Clarenti\Data\Project\ML\Python\Basic\DatasetFile', zips)

# =============================================================================
# Read Dataset and Label the Training Datasets 
# =============================================================================

def write_to_csv(directory, csv):
    data = []
    for folder in os.listdir(directory): # >>> [0, 1]
        for filenames in os.listdir(os.path.join((directory + "/"+ folder))): # >>> img001.jpg
            items = [filenames, folder]
            data.append(items)

    np.savetxt(("".join(csv + "/" + 'train_labels.csv')), data, delimiter=',', fmt='%s')


base_dir = "C:\\Clarenti\\Data\\Project\\ML\\Python\\Basic\\DatasetFile\\driver_behaviour\\"
train_dir = base_dir + "train"
validation_dir = base_dir + "test"
csv_dir = base_dir + "csv"

# write_test_to_csv(validation_dir, csv_dir)

# =============================================================================
# DATA AUGMENTATION (Image, Audio, Video, Text)
# Since Neural Network needs huge training dataset to recognize the image,
    # This augmentation is a technique that can be used to artificially expand the size of a training set, 
        # by creating modified data from the existing one. 
        # It's a good practice to use DA if you want to prevent overfitting.
    
    # In this augmentation process, you are able to try doing as following (image):
        # Geometric transformation: Crop, Rotate, Flip, Zoom, ...
        # Kernel filters: Sharpen or blur the images.
        # Color transformation: Change RGB color channels or intensify any color.
        # Erasing: Erase a part of the image.
        
# https://neptune.ai/blog/data-augmentation-in-python
# =============================================================================

def convert_img_to_arr(imgdir):
    X_train = []
  
    for folders in os.listdir(imgdir): # >>> [0, 1]
        for imgs in os.listdir((imgdir + "//" + folders)):
            img = cv2.imread(os.path.join((os.path.join(imgdir, folders)), imgs))
            imgarr = np.asarray(img, dtype="int32")
            X_train.append(imgarr)
        
    return X_train
    
# X_train = convert_img_to_arr(train_dir)

def augmentation(imagedir):
    # try to add flipped img, filter
    train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)
   
    test_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True, 
                                  vertical_flip=True,
                                  fill_mode='nearest')

    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(320, 240),  # All images will be resized to 150x150
        batch_size=25, # the number of training examples in one forward/backward pass 'per batch'
                       # the higher batch size the more memory you need
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary'
    )
    # print(train_generator)
    
    # test_df = pd.read_csv((csv_dir + "//test_df.csv")) -> for flow_from dataframe
    test_generator = test_datagen.flow_from_directory(
        validation_dir, 
        target_size=(320,240),
        batch_size=25,
        class_mode='binary'
    )
    
    return train_generator, test_generator

train_generator, test_generator = augmentation(train_dir)
# =============================================================================
# VGG16
    # CNN Architecture
    # object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with 92.7% accuracy
# =============================================================================

def build_model():
    vgg = models.Sequential()
    
    vgg.add(VGG16(include_top=False, input_shape=(320,240, 3), weights='imagenet'))
    for layer in vgg.layers:
        layer.trainable = False
        
    vgg.add(Flatten())
    vgg.add(Dense(units=128, activation="relu"))
    vgg.add(Dropout(0.2))
    vgg.add(Dense(units=128, activation="relu"))
    vgg.add(Dropout(0.3))
    vgg.add(Dense(units=2, activation="softmax")) #2 kategori
    
    vgg.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])
    
    vgg.summary()
    
    return vgg

# =============================================================================
# Train Model
# =============================================================================


checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')

def train():
    # Create own train
    model = build_model()
    # model.fit(X_train)
    model.fit(train_generator, 
              batch_size=10,
              # steps_per_epoch=2,  # n images = batch_size * steps
              epochs=3,
              validation_data=train_generator,
              validation_steps=1,
              callbacks=[checkpoint, early]
    )

    return model

# =============================================================================
# Test Perfomance of the Model
# =============================================================================

def predict():
    # create own predict
    model = train()
    test_generator.reset()
    y_pred = model.predict(test_generator)
        
    predicted_class_indices=np.argmax(y_pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions})
    results.to_csv((csv_dir + "//" + "results_sample.csv"),index=False)


predict()

    