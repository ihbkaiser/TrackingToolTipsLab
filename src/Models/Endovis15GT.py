import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Mask import Mask
import numpy as np
class Endovis15GT:
    def __init__(self, gt_mask):
        self.gt_mask = gt_mask
    def getBoundingBox(self, image):
        n = len(self.gt_mask)
        abstract_bb = np.zeros((n, 4))
        for i in range(n):
            x1, y1, x2, y2 = self.gt_mask[i].getBoundingBox()
            try:
                abstract_bb[i] = np.array([x1, y1, x2, y2])
            except:
                print("Error in getBoundingBox")
                print("x1, y1, x2, y2: ", x1, y1, x2, y2)
                print("abstract_bb: ", abstract_bb)
                print("i: ", i)
                print("n: ", n)
        return abstract_bb
    def getSkeletons(self, image):
        skeletons = []
        for m in self.gt_mask:
            mask = np.array(m.getSkeleton(), dtype=np.uint8)
            skeletons.append(mask)
        return np.stack(skeletons)
    def getMasks(self, image):
        return self.gt_mask