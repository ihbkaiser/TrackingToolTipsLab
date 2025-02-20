from tqdm import tqdm
from multiprocessing import Pool
import xml.etree.ElementTree as ET
from Video import Video
from Mask import Mask
from Models.Endovis15GT import *
from Models.YOLOModel import *
import os
class LoadEndovis15:
    def __init__(self, id):
        self.vid_path = f'../data/endovis15/Segmentation_Robotic_Testing/Segmentation/Dataset{str(id)}/Video.avi'
        self.gt_path = f'../data/endovis15/Segmentation_Robotic_Testing_GT/Dataset{str(id)}'
        # list all ".avi" files in the path
        self.vid = Video(self.vid_path)
        self.gt = [Video(self.gt_path + "/" + f) for f in os.listdir(self.gt_path) if f.endswith('.avi')]
        self.tool_tip = []
        self.get_tool_tip()
    def get_tool_tip(self):
        # list all ".xml" file in the path
        xmls = [self.gt_path + "/" + f for f in os.listdir(self.gt_path) if f.endswith('.xml')]
        assert len(xmls) == len(self.gt)
        for i in range(len(xmls)):
            image_info_list = []
            tree = ET.parse(xmls[i])
            root = tree.getroot()
            for img in root.iter('image'):
                image_info = [img.attrib['id'], img.attrib['name']]
                points_list = [point for points in img.iter('points') for point in points.attrib['points'].split(';')]
                points_int = []
                for point in points_list:
                    point = point.split(',')
                    point = [int(float(point[1])), int(float(point[0]))]
                    points_int.append(point)
                image_info.extend(points_int)
                image_info_list.append(image_info)
                image_info_list = sorted(image_info_list, key=lambda x: int(x[1].split('.')[0]))
            self.tool_tip.append(image_info_list)
        
      
    def process_frame(args):
        idx, frame, gt_list = args
        mask = []
        for gt in gt_list:
            mask.append(Mask(gt.getFrame(idx).image))
        model_here = Endovis15GT(mask)
        frame.setAll(model_here)
        return frame
    def process(self, multiprocessing = True, method = "YOLOv11Model"):
        if not multiprocessing:
            self.vid.setFrames()
            frames = self.vid.frames
            # frames = frames[0:5]   #for testing only
            sample_frame = []
            idx = 0
            for frame in tqdm(frames):
                mask = []
                for gt in self.gt:
                    mask.append(Mask(gt.getFrame(idx).image))
                if method == "Endovis15GT":
                    model_here = Endovis15GT(mask)
                elif method == "YOLOv9Model":
                    model_here = YOLOModel("Models/yolo_weights/endovis15.pt")
                elif method == "YOLOv11Model":
                    model_here = YOLOModel("Models/yolo_weights/endovis15-2025.pt")
                # model_here = YOLOModel("Models/yolo_weights/endovis15.pt")
                #model_here = LabModel()
                # model_here = YOLOModel("yolov9c-best.pt")
                ## if u want to use LabModel, set model_here = LabModel()
                frame.setAll(model_here)
                sample_frame.append(frame)
                idx = idx + 1

            return sample_frame
        if multiprocessing:
            #TODO: how to read frames simultaneously
            self.vid.setFrames()
            frames = self.vid.frames
            sample_frame = []
            with Pool() as p:
                args = [(idx, frame, self.gt) for idx, frame in enumerate(frames)]
                sample_frame = list(tqdm(p.imap(self.process_frame, args), total = len(frames)))
            return sample_frame