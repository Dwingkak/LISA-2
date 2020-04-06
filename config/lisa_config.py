# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:40:18 2020

@author: Kdwing
"""

import os

# initialize the base path for the LISA dataset
BASE_PATH = "dataset"

# build the path to the annotation file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

# the path to the training and testing record files
TRAIN_RECORD = os.path.sep.join([BASE_PATH, 
                                 "records", "training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, 
                                "records", "testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, 
                                 "records", "classes.pbtxt"])

# test split size
TEST_SIZE = 0.25

# class label dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}



















