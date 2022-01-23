# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 13:13:29 2022

@author: Ding Qi
"""
import cv2
import time 
import numpy as np
"""
image=cv2.imread("./data/lane.png")  #要注意这种相对路径的写法
  
#cv2.destoryAllWindows()   #关闭窗口

#二值化处理，将彩色图片转换成灰度图

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#这里刚刚报错出现问题，通过stackoverflow对问题进行解决

#cv2.imshow("lane",gray_image)   #展示图像
#cv2.waitKey()  
#截至到这里完成彩色图到灰度图的转变

blured_image = cv2.GaussianBlur(gray_image,(5,5),0)  #这里的参数可以自行调整
#第二个参数表示模糊半径，第三个参数代表高斯函数标准差，半径越大标准差越大，图片越模糊

time.sleep(3)
#cv2.destroyAllWindows()
#cv2.imshow("blured_image",blured_image)   #展示高斯模糊后的图像

canny = cv2.cv2.Canny(blured_image,250,300)  #canny边缘检测
#cv2.imshow("canny",canny)

#提取所得到的图像有很多是与道路无关的区域，选择ROI区域提取车道线所在区域进行处理

roi_range = np.array([[(10,550),(433,332),(500,333),(949,540)]],dtype = np.int32)
#我发现这几个点的顺序不同也是会影响结果的  注意0是黑，255是白
mask = np.zeros_like(canny)    #复制一个和canny图像大小一样的叠加矩阵
cv2.fillPoly(mask,roi_range,255)   #设置roi区域的像素值为255，其他区域为0
img_masked = cv2.bitwise_and(canny,mask)    #将canny图像和叠加图像求并，ROI区域外的部分都变为0（黑）
#cv2.imshow("mixed_picture",img_masked)  

#直线检测提取车道（通过霍夫变换）

lines = cv2.HoughLinesP(img_masked,1,np.pi/180,15,25,20) #这里的几个参数要看下，自己要调整的
#我发现调参是一件非常麻烦的事情，或许以后需要写代码进行自动调参判断等，计算机科学大有可为
#能通过调包完成自己的任务已经非常厉害，自己研究算法这种事情还是留给大神来做
for line in lines:
	for x1,y1,x2,y2 in line:
		cv2.line(img_masked,(x1,y1),(x2,y2),255,8)  #应该是对线段进行加粗
#cv2.imshow('img_masked',img_masked)
#cv2.waitKey()
#cv2.destroyAllWindows()      #就是左侧三条线，右侧一条线
#opencv默认原点位于图像左上角顶点，所以左边线斜率为负，右边线斜率为正

positive_slop_intercept = []    #左边线点构成直线的斜率和截距
negative_slop_intercept = []  #右边线点构成直线的斜率和截距
for line in lines:
	for x1,y1,x2,y2 in line:
		slop = np.float((y2-y1))/np.float((x2-x1))  #计算斜率
		if slop < 0:                 
			positive_slop_intercept.append([slop,y1-slop*x1]) #根据点的坐标和斜率计算截距
		elif slop > 0 :
			negative_slop_intercept.append([slop,y1-slop*x1])

#numpy是用来处理数据的库，某些意义上我觉得可以认为它是万能库

legal_slop=[]
legal_intercept=[]
slopes=[pair[0] for pair in positive_slop_intercept]
slop_mean = np.mean(slopes)                      #斜率的均值
slop_std = np.std(slopes)                        #斜率的标准差
for pair in positive_slop_intercept :
	if pair[0] - slop_mean < 3 * slop_std:      #挑选出平均值3个标准差误差范围内的斜率和截距
		legal_slop.append(pair[0])
		legal_intercept.append(pair[1])
if not legal_slop:                       #应该是非空判断，如果没有合理范围内的斜率，则使用原始斜率，最终的斜率就是均值
	legal_slop = slopes
	legal_intercept = [pair[1] for pair in positive_slop_intercept ]  #pair中的0和1分别取斜率和截距
slop_l = np.mean(legal_slop)
intercept_l = np.mean(legal_intercept)

slop_r=negative_slop_intercept[0][0]  #左边的线只有一个，故直接索取
intercept_r=negative_slop_intercept[0][1]


ymin=345
ymax=550               
xmin_r= int((ymin - intercept_r)/slop_r)    #计算出左侧车道线的两个端点
xmax_r= int((ymax-intercept_r)/slop_r)

xmin_l= int((ymin - intercept_l)/slop_l)    #计算出左侧车道线的两个端点
xmax_l= int((ymax-intercept_l)/slop_l)

#进行图像的叠加
only_lane= np.zeros_like(image)  #构建同样大小的黑色图，画上车道线，与原图进行重叠
cv2.line(only_lane,(xmin_l,ymin),(xmax_l,ymax),(255,255,0),8)
cv2.line(only_lane,(xmin_r,ymin),(xmax_r,ymax),(255,0,255),8)
#cv2.imshow("only_lane",only_lane)
#imgae为原始图像，line_image为车道标记图像
final_image = cv2.addWeighted(image,0.8,only_lane,1,0)  #应该是像素值的叠加，黑色为0不影响原图像
#cv2.imshow("final_image",final_image)  """

#视频处理
from moviepy.editor import *


def process_an_image(image):    #这里的自变量直接是图片，而非图片的地址
   # image=cv2.imread(path)  #要注意这种相对路径的写法
      
    #cv2.destoryAllWindows()   #关闭窗口
    
    #二值化处理，将彩色图片转换成灰度图
    
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #这里刚刚报错出现问题，通过stackoverflow对问题进行解决
    
    #cv2.imshow("lane",gray_image)   #展示图像
    #cv2.waitKey()  
    #截至到这里完成彩色图到灰度图的转变
    
    blured_image = cv2.GaussianBlur(gray_image,(5,5),0)  #这里的参数可以自行调整
    #第二个参数表示模糊半径，第三个参数代表高斯函数标准差，半径越大标准差越大，图片越模糊
    
    time.sleep(3)
    #cv2.destroyAllWindows()
    #cv2.imshow("blured_image",blured_image)   #展示高斯模糊后的图像
    
    canny = cv2.cv2.Canny(blured_image,250,300)  #canny边缘检测
    #cv2.imshow("canny",canny)
    
    #提取所得到的图像有很多是与道路无关的区域，选择ROI区域提取车道线所在区域进行处理
    
    roi_range = np.array([[(10,550),(433,332),(500,333),(949,540)]],dtype = np.int32)
    #我发现这几个点的顺序不同也是会影响结果的  注意0是黑，255是白
    mask = np.zeros_like(canny)    #复制一个和canny图像大小一样的叠加矩阵
    cv2.fillPoly(mask,roi_range,255)   #设置roi区域的像素值为255，其他区域为0
    img_masked = cv2.bitwise_and(canny,mask)    #将canny图像和叠加图像求并，ROI区域外的部分都变为0（黑）
    #cv2.imshow("mixed_picture",img_masked)  
    
    #直线检测提取车道（通过霍夫变换）
    
    lines = cv2.HoughLinesP(img_masked,1,np.pi/180,15,25,20) #这里的几个参数要看下，自己要调整的
    #我发现调参是一件非常麻烦的事情，或许以后需要写代码进行自动调参判断等，计算机科学大有可为
    #能通过调包完成自己的任务已经非常厉害，自己研究算法这种事情还是留给大神来做
    line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    if lines is None:
        return line_img
    
    for line in lines:
    	for x1,y1,x2,y2 in line:
    		cv2.line(img_masked,(x1,y1),(x2,y2),255,8)  #应该是对线段进行加粗
    #cv2.imshow('img_masked',img_masked)
    #cv2.waitKey()
    #cv2.destroyAllWindows()      #就是左侧三条线，右侧一条线
    #opencv默认原点位于图像左上角顶点，所以左边线斜率为负，右边线斜率为正
    
    positive_slop_intercept = []    #左边线点构成直线的斜率和截距
    negative_slop_intercept = []  #右边线点构成直线的斜率和截距

    for line in lines:
    	for x1,y1,x2,y2 in line:
    		slop = np.float((y2-y1))/np.float((x2-x1))  #计算斜率
    		if slop < 0:                 
    			positive_slop_intercept.append([slop,y1-slop*x1]) #根据点的坐标和斜率计算截距
    		elif slop > 0 :
    			negative_slop_intercept.append([slop,y1-slop*x1])
                
    if (len(positive_slop_intercept)==0) or (len(negative_slop_intercept)==0):
        return image
    """最后一直debug就是这句话，没有考虑到数据缺失NAN的情况，我们不能保证每一帧都能生成车道线 """
    
    #numpy是用来处理数据的库，某些意义上我觉得可以认为它是万能库
    
    legal_slop=[]
    legal_intercept=[]
    slopes=[pair[0] for pair in positive_slop_intercept]
    slop_mean = np.mean(slopes)                      #斜率的均值
    slop_std = np.std(slopes)                        #斜率的标准差
    for pair in positive_slop_intercept :
    	if pair[0] - slop_mean < 3 * slop_std:      #挑选出平均值3个标准差误差范围内的斜率和截距
    		legal_slop.append(pair[0])
    		legal_intercept.append(pair[1])
    if not legal_slop:                       #应该是非空判断，如果没有合理范围内的斜率，则使用原始斜率，最终的斜率就是均值
    	legal_slop = slopes
    	legal_intercept = [pair[1] for pair in positive_slop_intercept ]  #pair中的0和1分别取斜率和截距
    slop_l = np.mean(legal_slop)
    intercept_l = np.mean(legal_intercept)
    
    slop_r=negative_slop_intercept[0][0]  #左边的线只有一个，故直接索取
    intercept_r=negative_slop_intercept[0][1]
    
    
    ymin=345
    ymax=550
    xmin_r= int((ymin - intercept_r)/slop_r) #计算出左侧车道线的两个端点
    xmax_r= int((ymax-intercept_r)/slop_r)
    
    xmin_l= int((ymin - intercept_l)/slop_l)    #计算出左侧车道线的两个端点
    xmax_l= int((ymax-intercept_l)/slop_l)
    
    #cannot convert float NaN to integer，这里报错，数据集里面的缺失值需要填充起来，避免各种出错
    #关键在于对NaN的处理
    
    #进行图像的叠加
    only_lane= np.zeros_like(image)  #构建同样大小的黑色图，画上车道线，与原图进行重叠
    cv2.line(only_lane,(xmin_l,ymin),(xmax_l,ymax),(255,255,0),8)
    cv2.line(only_lane,(xmin_r,ymin),(xmax_r,ymax),(255,0,255),8)
    #cv2.imshow("only_lane",only_lane)
    #imgae为原始图像，line_image为车道标记图像
    final_image = cv2.addWeighted(image,0.8,only_lane,1,0)  #应该是像素值的叠加，黑色为0不影响原图像
    #cv2.imshow("final_image",final_image)
    return final_image
    
output="./movie/with_lane.mp4"
clip = VideoFileClip("./movie/cv2_white_lane.mp4")
out_clip =clip.fl_image(process_an_image)
out_clip.write_videofile(output, audio=False)