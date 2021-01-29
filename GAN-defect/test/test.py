# coding=UTF-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'/home/lyl/object detection/GAN-defect/test/t.bmp',1)
# plt.imshow(img,)
# plt.show()
# 利用cv2.threshhold()函数进行简单阈值分割，第一个参数是待分割图像，第二个参数是阈值大小
# 第三个参数是赋值的像素值，第四个参数是阈值分割方法
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
# print(thresh1.shape)
# plt.imshow(thresh1,)
# plt.show()
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# images = [ thresh1]
# plt.imshow(thresh1, 'gray')
# plt.savefig("f.png")
# for i in xrange(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),plt.savefig("f.png")
#     # plt.title(titles[i])
#     plt.xticks(), plt.yticks([])  # 显示坐标轴，如为空，则无坐标轴

# f_img=cv2.imread('/home/lyl/object detection/GAN-defect/test/f.png',0)
# print (img.shape,thresh1.shape,)
subtracted=cv2.subtract(img,thresh1)
print(subtracted)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)
