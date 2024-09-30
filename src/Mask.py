from ROI import ROI
import cv2
import numpy as np
from shapely.geometry import Polygon
class Mask:
    def __init__(self, mask, threshold = 10, white_pixel = 255, crop=False):
        # check if mask is bool matrix
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        if len(mask.shape) == 3:
            grayscale_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = mask
        grayscale_image[grayscale_image > threshold] = white_pixel
        self.mask = grayscale_image
        if crop:
            x1 = ROI[0][0]
            x2 = ROI[0][1]
            y1 = ROI[1][0]
            y2 = ROI[1][1]
            self.mask = self.mask[y1:y2, x1:x2]

    def isEmpty(self):
        return np.count_nonzero(self.mask) == 0
    def getSkeleton(self):
        return cv2.ximgproc.thinning(self.mask)
    def getBoundingBox(self):
        if self.isEmpty():
            raise ValueError("Mask is empty")
        else:
            # for all pixels in the mask, if the pixel different than 0, update min_x, max_x, min_y, max_y
            min_x = 999999
            min_y = 999999
            max_x = -999999
            max_y = -999999
            for y in range(len(self.mask)):
                for x in range(len(self.mask[y])):
                    if self.mask[y][x] != 0:
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
            return (min_x, min_y, max_x, max_y)
    def toPolygon(self):
        ## https://github.com/ultralytics/ultralytics/issues/3085#issuecomment-1643514634
        mask = self.mask.astype(bool)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        normalized_polygons = []
        for contour in contours:
            try:
                polygon = contour.reshape(-1, 2).tolist()
                normalized_polygon = [[round(coord[0] / mask.shape[1] , 4), round(coord[1] / mask.shape[0] , 4)] for coord in polygon]
        
                polygon_shapely = Polygon(polygon)
                simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
                polygons.append(simplified_polygon)

                normalized_polygons.append(Polygon(normalized_polygon))
          

            except Exception as e:
                pass
        
        return polygons, normalized_polygons