# lane_detection
Basic approach of lane detection using python , totally three py files and two videos

基本的车道线识别
导入一段视频，通过moivepy进行抽帧，函数处理，组合
其中lane_detection.py为主文件
ROI——location_return 用于主文件中确定ROI区域时获取像素点坐标，需要单独运行

reference.py为参考别人的github代码的原文件
https://github.com/yajian/self-driving-car

tips：
1.canny边缘提取时的参数需要调整
2.要考虑到单张图片处理时未检测到车道线的情况，否则对视频处理中途容易报错
3.最好将各个功能封装成函数，便于复用，利用__main__的形式，大段代码难以debug
4.理论上还可以加入对曲线再次处理（例如滤去噪声，平滑处理等）
5.最后的addweight处在未检测到车道线时视频亮度会变化，此处未作处理

date
2022/1/23  22:00 PM
