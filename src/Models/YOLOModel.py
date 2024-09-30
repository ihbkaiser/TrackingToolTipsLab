from ultralytics import YOLO
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Mask import Mask
class YOLOModel:
    def __init__(self, path_to_model):
        self.model = YOLO(path_to_model)
    def getBoundingBox(self, image):
        res = self.model(image)
        return res[0].boxes.xyxy.cpu().numpy()
    def getMasks(self, image):
        res = self.model(image)
        if res[0].masks is None:
            return []
        masks = res[0].masks.data.cpu().numpy()
        lst = []
        for i in range(len(masks)):
            mask = masks[i].astype(np.uint8) * 255
            
            lst.append(Mask(mask))
        return lst
        