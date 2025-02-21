#coding:utf-8
# -*- coding: utf-8 -*-
# @Author  : NaiChuan
# @Time    : 2024/8/28 13:45
# @File    : train.py
# @Software: PyCharm

from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    # model = YOLO('D:/Eggg/CVR EGG 4.v2i.yolov11/YOLOv8_MobileNetv4_ultralytics/ultralytics/cfg/models/v8/yolov8s-mobilenetv4.yaml')
    model = YOLO(r'D:\Eggg\CVR EGG 4.v2i.yolov11\weight18\weights\best.pt')

    # model.load('yolov8n.pt') # loading pretrain weights
    # print(model)
    teacher_model = YOLO(r"C:\Users\ADMIN\Downloads\yolo11x-seg.pt")

    results=model.train(data=r'D:/Eggg/CVR EGG 4.v2i.yolov11/data.yaml',
                        epochs=150,
                        imgsz=640,
                        val=True,
                        patience=0,
                        device=0,
                        verbose=True,
                        name='D:/Eggg/CVR EGG 4.v2i.yolov11/weight',
                        )