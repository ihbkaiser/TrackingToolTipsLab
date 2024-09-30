class LabModel:
    def __init__(self):
        self.model = InferenceHTTPClient(api_url="https://outline.roboflow.com",api_key="NzQOXBqQrzYP4mT38Ksg")
        self.name = "sugical-tool-iszjm/1"
    def getBoundingBox(self, image):
        '''
        Return n x 4 array containing the bounding boxes coordinates in format [x1, y1, x2, y2], which x1<=X<=x2, y1<=Y<=y2
        '''
        result = self.model.infer(image, self.name)
        detections = sv.Detections.from_roboflow(result)
        return detections.xyxy
    def getMasks(self, image):
        '''
        Return a Python list of Mask(s)
        '''
        result = self.model.infer(image, self.name)
        detections = sv.Detections.from_inference(result)
        mask = detections.mask
        mask_list = []
        for i in range(len(mask)):
            print(mask[i])
            mask_list.append(Mask(mask[i]))
        return mask_list