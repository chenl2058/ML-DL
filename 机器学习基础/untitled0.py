
import cv2

#声明变量，负责人脸检测
face_cascade=cv2.CascadeClassifier('./OpenCV/cascades/haarcascade_frontalface_default.xml')
#声明变量，负责眼睛检测
eye_cascade=cv2.CascadeClassifier('./OpenCV/cascades/haarcascade_eye.xml')

cv2.namedWindow("Image")

camera=cv2.VideoCapture(0)
#frame=camera.read()[1]#参数：布尔值，表示是否成功读取帧；读到的帧
while (True):
    k=cv2.waitKey(1) 
    if k == 27:break

    ret,frame=camera.read()#参数：布尔值，表示是否成功读取帧；读到的帧
    
    if ret==False:
        print("没有捕获到视频，ESC键退出")
        continue
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #进行实际的人脸检测
    #参数：图像、人脸过程中每次迭代时图像的压缩率、每个人脸矩形保留近邻数目的最小值
    #返回：人脸矩形数目
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:#依次提取出每个人脸
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#直接在图像上绘制矩形
        #在矩形区域检测眼睛
        roi_gray=gray[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.03,5,0,(40,40))
        for (ex,ey,ew,eh) in eyes:#依次提取出眼睛
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)#直接再图像上绘制矩形
    
    cv2.imshow('Image',frame)

    #ret,frame=camera.read()#参数：布尔值，表示是否成功读取帧；读到的帧

camera.release()
cv2.destroyAllWindows()
