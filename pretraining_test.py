# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:30:03 2020

@author: Kdwing
"""
from config import lisa_config as config
import os
from PIL import Image
import cv2
import random

D = {}
rows = open(config.ANNOT_PATH).read().strip().split("\n")
rows = random.choices(rows[1:], k = 100)
for row in rows:
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

keys = list(D.keys())
print("total number of images are: {}".format(len(keys)))
for k in keys:
    pilImage = Image.open(k)
    (w, h) = pilImage.size[:2]
    for (label, (startX, startY, endX, endY)) in D[k]:
        xMin = startX / w
        xMax = endX / w
        yMin = startY / h
        yMax = endY / h
     
    # load input image from disk and denormalize the bbox coordinates
        image = cv2.imread(k)
        startX = int(xMin * w)
        startY = int(yMin * h)
        endX = int(xMax * w)
        endY = int(yMax * h)
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        # show the output image
        cv2.imshow(label, image)
        cv2.waitKey(0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        