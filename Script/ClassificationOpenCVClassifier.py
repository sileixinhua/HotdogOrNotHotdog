# -*- coding:utf-8 -*-  
# coder:橘子派_司磊
# 2017年11月5日 17点34分
# 训练你自己的类OPENCV HAAR CLASSIFIER分类器
# TRAIN YOUR OWN OPENCV HAAR CLASSIFIER
# http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
# 热狗图片一共11582张
# 火锅照片一共1043张
# Make Your Own Haar Cascades
# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
# 处理后的热狗图片一共11488张
# '..\Data\HotdogResize'
# 处理后的火锅照片一共1016张
# '..\Data\HotpetResize'

# 结果热狗的图片一张都没有用到
# 测试的时候记得用手机打开图片watch5050.jpg
# 笔记本运算的，所以训练集就用黑白的，样本也少，测试的时候多上下左右运动下
# 这次用的opencv3.6的cascade处理流程，的确比tensorflow之类方便点，但是要求的样本要大
# 效果没有tensorflow好，小的需求和在移动设备上可以用这种方法

# 代码说明在下方主函数

import cv2
import os
import numpy as np

def GrayResizePictures():
    if not os.path.exists('..\Data\HotdogResize'):
        os.makedirs('..\Data\HotdogResize')
    if not os.path.exists('..\Data\HotpetResize'):
        os.makedirs('..\Data\HotpetResize')   

    for HotpetRoot,HotpetDirs,HotdogFiles in os.walk('..\Data\Hotdog'):  
        #print(HotpetRoot)
        #print(HotpetDirs)
        #print(HotdogFiles)
        j = 0
        for i in HotdogFiles:
            j=j+1
            img = cv2.imread('..\Data\Hotdog\\'+str(HotdogFiles[j]),cv2.IMREAD_GRAYSCALE)
            if img is  None:
                continue
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite('..\Data\HotdogResize\\'+str(HotdogFiles[j]),resized_image)
    for HotpetRoot,HotpetDirs,HotpetFiles in os.walk('..\Data\Hotpet'):  
        #print(HotpetRoot)
        #print(HotpetDirs)
        #print(HotpetFiles)
        j = 0
        for i in HotpetFiles:
            j=j+1
            img = cv2.imread('..\Data\Hotpet\\'+str(HotpetFiles[j]),cv2.IMREAD_GRAYSCALE)
            if img is  None:
                continue
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite('..\Data\HotpetResize\\'+str(HotpetFiles[j]),resized_image)

def Create_pos_n_neg():
    for file_type in ['HotpetResize']:
    
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'HotpetResize':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

def Dectctive():
    face_cascade = cv2.CascadeClassifier('C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

    watch_cascade = cv2.CascadeClassifier('C:\\Code\\HotdogOrNotHotdog\\Script\\cascade_hotdog.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        watches = watch_cascade.detectMultiScale(gray, 10, 10)
        # 这里参数可改成 5
        # detectMultiScale()
        # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
        # minSize – Minimum possible object size. Objects smaller than that are ignored.
        # maxSize – Maximum possible object size. Objects larger than that are ignored.
        
        for (x,y,w,h) in watches:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # 1.先执行GrayResizePictures()和Create_pos_n_neg()函数
    # 2.再当前目录下用cmd或者terminal执行
    # opencv_createsamples两个命令，创建和整理训练集
    # 3.执行opencv_traincascade，根据训练集的数量再data文件夹下不同生成的结果
    # 4.然后再执行一次opencv_traincascade，记得把最后数字改成data文件夹下生成的数字
    # 5.执行Dectctive()函数

    # GrayResizePictures()
    # Create_pos_n_neg()

    # opencv_createsamples -img watch5050.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1106
    # opencv_createsamples -info info/info.lst -num 1106 -w 20 -h 20 -vec positives.vec
    # mkdir data

    # opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1000 -numNeg 500 -numStages 6 -w 20 -h 20
    # opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1000 -numNeg 500 -numStages 5 -w 20 -h 20

    # 这里主要的方法就是用一张图片做正例和很多张反例图片，把整理图片缩小加入到反例图片中
    # 就形成了正例，然后traincascade

    Dectctive()