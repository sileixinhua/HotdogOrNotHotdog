# -*- coding:utf-8 -*-  
# coder:橘子派_司磊
# 2017年10月31日 时间
# 用OpenCV来识别图片是否是热狗
# 热狗图片一共11582张
# 火锅照片一共1043张
# 效果十分不理想！！

import cv2
import numpy as np 

# 计算单通道的直方图的相似值 
def calculate(image1,image2): 
	hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
	hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
	# 计算直方图的重合度 
	degree = 0
	for i in range(len(hist1)): 
		if hist1[i] != hist2[i]: 
			degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
		else: 
			degree = degree + 1
	degree = degree/len(hist1) 
	return degree 
  
# 通过得到每个通道的直方图来计算相似度 
def classify_pHash(image1,image2): 
	image1 = cv2.resize(image1,(32,32)) 
	image2 = cv2.resize(image2,(32,32)) 
	gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
	gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
	# 将灰度图转为浮点型，再进行dct变换 
	dct1 = cv2.dct(np.float32(gray1)) 
	dct2 = cv2.dct(np.float32(gray2)) 
	# 取左上角的8*8，这些代表图片的最低频率 
	# 这个操作等价于c++中利用opencv实现的掩码操作 
	# 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分 
	dct1_roi = dct1[0:8,0:8] 
	dct2_roi = dct2[0:8,0:8] 
	hash1 = getHash(dct1_roi) 
	hash2 = getHash(dct2_roi) 
	return Hamming_distance(hash1,hash2) 
  
# 输入灰度图，返回hash 
def getHash(image): 
	avreage = np.mean(image) 
	hash = [] 
	for i in range(image.shape[0]): 
		for j in range(image.shape[1]): 
			if image[i,j] > avreage: 
				hash.append(1) 
			else: 
				hash.append(0) 
	return hash
  
# 计算汉明距离 
def Hamming_distance(hash1,hash2): 
	num = 0
	for index in range(len(hash1)): 
		if hash1[index] != hash2[index]: 
			num += 1
	return num 

if __name__ == '__main__': 
	img1 = cv2.imread('../Data/Hotdog/1.jpg') 
	img2 = cv2.imread('../Data/Hotpet/1.jpg')
	# 取一个热狗和火锅的图做对比，效果为相似度为31，可见效果十分不好
	degree = classify_pHash(img1,img2) 
	print(degree) 
	cv2.waitKey(0)
