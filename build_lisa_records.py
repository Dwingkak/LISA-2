# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:57:27 2020

@author: Kdwing
"""
from config import lisa_config as config
from tools.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os

def main(_):
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
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))
        
        # exclude label not included in the classes dictionary
        if label not in config.CLASSES:
            continue
        
        # path to the input image
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        # create a list to be stored according to imagePath
        b = D.get(p, [])
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b
        
    # create training and testing splits from data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
                                             test_size = config.TEST_SIZE,
                                             random_state = 42)
    
    # initialize data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
        ]
    
    # loop over the datasets
    for (dType, keys, outputPath) in datasets:
        # initialize Tensorflow writer and initialize 
        # the total number of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0
        
        # loop over the keys in the current set
        for k in keys:
            # load the input image from disks as a Tensorflow object
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)
            
            # load the image from disk again as PIL object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]
            
            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            
            # initialize annotation object used to store
            # bounding box coordinates and labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h
            
            # loop over bounding boxes and label for an image
            for (label, (startX, startY, endX, endY)) in D[k]:
                # Tensorflow assumes all bounding boxes in the
                # range of [0, 1] thus need to rescaling
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h
                
                # update bounding boxes and labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)
                
                # increment total number of examples
                total += 1
            
            # encode the data point attributes using the Tensorflow
            # helper functions
            features = tf.train.Features(feature = tfAnnot.build())
            example = tf.train.Example(features = features)
            
            # add the example to the writer
            writer.write(example.SerializeToString())
            
        # close the writer and print diagnostic information
        writer.close()
        print("[INFO] {} examples saved for {}".format(total, dType))
        
# check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()
            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
