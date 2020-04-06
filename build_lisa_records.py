# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:57:27 2020

@author: Kdwing
"""
from config import lisa_config as config
from tools.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import image
import tensorflow as tf
import os

def main():
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")
    
    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)
    f.close()
    
    # initialize a data dictionary used to map each image
    # filename to all bounding boxes associated with the
    # image, then load the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")
    for row in rows[1:]:
        























