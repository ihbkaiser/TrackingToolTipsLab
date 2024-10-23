class LabModel:
    def __init__(self):
        # replace xxxx0123456789 with your own roboflow API key
        self.model = InferenceHTTPClient(api_url="https://outline.roboflow.com",api_key="xxxx0123456789")
        self.name = "sugical-tool-iszjm/1"
    def getBoundingBox(self, image):
        '''
        Return n x 4 array containing the bounding boxes coordinates in format [x1, y1, x2, y2], which x1<=X<=x2, y1<=Y<=y2
        '''
        result = self.model.infer(image, self.name)
        detections = sv.Detections.from_inference(result)
        return detections.xyxy
    def getMasks(self, image):
        '''
        Return a Python list of Mask(s)
        '''
        result = self.model.infer(image, self.name)
        detections = sv.Detections.from_inference(result)
        mask = detections.mask
        mask_list = []
        if mask is not None:
          for i in range(len(mask)):
              mask_list.append(Mask(mask[i]))
        return mask_list