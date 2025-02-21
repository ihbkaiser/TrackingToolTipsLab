#coding:utf-8
# -*- coding: utf-8 -*-
# @Author  : NaiChuan
# @Time    : 2024/8/28 13:45
# @File    : train.py
# @Software: PyCharm

from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    teacher_model = RTDETR("D:/Eggg/CVR EGG 4.v2i.yolov11/rtdetr-x.pt")

    model = YOLO('D:/Eggg/CVR EGG 4.v2i.yolov11/YOLOv8_MobileNetv4_ultralytics/ultralytics/cfg/models/v8/yolov8s-mobilenetv4.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # print(model)
    results=model.train(data=r'D:/Eggg/CVR EGG 4.v2i.yolov11/data.yaml',
                        epochs=150,#训练轮数
                        imgsz=640,#输入图像大小
                        val=True,#是否进行验证
                        teacher=teacher_model.model,
                        distillation_loss="cwd",
                        # lr0=0.012,#学习率设置
                        patience=0,#为0是取消早停，0-300是设置早停
                        device=0,#运行设备
                        # batch=64,#当模型较大的时候不设置batch，让它默认防止溢出
                        verbose=True, #看到更多训练信息
                        name='D:/Eggg/CVR EGG 4.v2i.yolov11/weight', #指定结果保存的文件夹名称,记得修改为自己的
                        # close_mosaic=0, #关闭moasic数据增强
                        deterministic=False #随机因子的作用，默认为True
                        )