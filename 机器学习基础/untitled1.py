# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:35:50 2020

@author: chenl
"""


import cv2
filename='./OpenCV/big/img_83.jpg'
def detect(filename):
    #声明变量，负责人脸检测
    face_cascade=cv2.CascadeClassifier('./OpenCV/cascades/haarcascade_frontalface_default.xml')
    
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #进行实际的人脸检测
    #参数：图像、人脸过程中每次迭代时图像的压缩率、每个人脸矩形保留近邻数目的最小值
    #返回：人脸矩形数目
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:#依次提取出每个人脸
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#直接再图像上绘制矩形
    #cv2.namedWindow('img_83')
    cv2.imshow('img_83',img)
    cv2.imwrite('./OpenCV/img_83.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
detect(filename)