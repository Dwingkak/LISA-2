# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:24:53 2020

@author: Kdwing
"""

from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature

class TFAnnotation:
    def __init__(self):
        # initialize bounding box and label lists
        self.xMins = []
        self.xMaxs = []
        self.yMins = []
        self.yMaxs = []
        self.textLabels = []
        self.classes = []
        self.difficult = []
        
        # initialize image, its spatial dimensions, encoding,
        # and file name
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None
    
    def build(self):
        # encode the attributes using respective Tensorflow
        # encoding function
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.encoding.encode("utf8"))
        image = bytes_feature(self.image)
        xMins = float_list_feature(self.xMins)
        xMaxs = float_list_feature(self.xMaxs)
        yMins = float_list_feature(self.yMins)
        yMaxs = float_list_feature(self.yMaxs)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)
        
        # construct Tensorflow compatible data dictionary
        data = {
            "image/height": h,
            "image/weight": w,
            "image/filename": filename,
            "image/source_id": filename,
            "image/encoded": image,
            "image/format": encoding,
            "image/object/bbox/xmin": xMins,
            "image/object/bbox/xmax": xMaxs,
            "image/object/bbox/ymin": yMins,
            "image/object/bbox/ymax": yMaxs,
            "image/object/bbox/text": textLabels,
            "image/object/bbox/label": classes,
            "image/object/difficult": difficult
            }
        
        return data





















