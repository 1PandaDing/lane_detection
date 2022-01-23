# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:04:10 2022

@author: Ding Qi
"""

"""打开图片，点击图片，得到像素点的坐标"""

import cv2
from PIL import Image
from pylab import *
im = array(Image.open("./data/lane.png"))
imshow(im)
print ('Please click 4 points')
x =ginput(2)
print ('you clicked:',x)
show()

#获取图像点坐标，以确定ROI区域的范围