from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0, 255, 0)
# 设置绿色

# 用matplotlib制作绘图函数
def show(image):
    plt.figure(figsize=(10, 10))
    # 设置图像大小
    plt.imshow(image, interpolation='nearest')
    # 绘制展示图像

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    # 叠加图像并设置图像透明度
    return img

def find_biggest_contour(image):
    image = image.copy()
    image,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 设置图像的一个拷贝，并绘制图像的轮廓
    # cv2.findContours为图像的轮廓检测

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    # cv2.drawContours 填充轮廓的颜色
    return biggest_contour, mask

def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    # 将对应的部分画圈
    return image_with_ellipse

def find_hotdog(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)
    
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    circled = circle_contour(overlay, big_strawberry_contour)
    show(circled)
    
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    
    return bgr

image = cv2.imread('1.jpg')
result = find_hotdog(image)
cv2.imwrite('2.jpg', result)