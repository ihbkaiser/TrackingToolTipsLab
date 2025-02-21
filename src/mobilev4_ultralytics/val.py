#coding:utf-8
# -*- coding: utf-8 -*-
# @Author  : NaiChuan
# @Time    : 2024/8/28 13:45
# @File    : train.py
# @Software: PyCharm

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/Eggg/CVR EGG 4.v2i.yolov11/weight13/weights/best.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    # print(model)
    results=model.val(data=r'D:/Eggg/CVR EGG 4.v2i.yolov11/test.yaml',
                        imgsz=640,#输入图像大小
                        # lr0=0.012,#学习率设置
                        device=0,#运行设备
                        # batch=64,#当模型较大的时候不设置batch，让它默认防止溢出
                        name='D:/Eggg/CVR EGG 4.v2i.yolov11/val', #指定结果保存的文件夹名称,记得修改为自己的
                        # close_mosaic=0, #关闭moasic数据增强
                        )