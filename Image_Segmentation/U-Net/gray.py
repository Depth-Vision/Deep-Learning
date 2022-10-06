import os
import cv2
import numpy as np
from skimage import io,transform,color


def convert_gray(f):                                            # 图片处理与格式化的函数
    rgb=io.imread(f)
    gray=color.rgb2gray(rgb)   
    dst=transform.resize(gray,(224,224))   
    return dst
    
def Gray(save_path,img_path,type=".png"):                       # 批量灰度化函数
    if os.path.exists(save_path):                               # 判断文件是否存在
        pass
    else:
        os.makedirs(save_path)                                  # 创建文件
    str=img_path+'/*'+ type
    coll = io.ImageCollection(str,load_func=convert_gray)
    for i in range(len(coll)):
        io.imsave(save_path + np.str(i) + '.png',coll[i])

def convert(f,**args):                                          # 图片处理与格式化的函数
    img=cv2.imread(f)                                           # 读取图片
    t,rst=cv2.threshold(img,0,255,cv2.THRESH_BINARY)            # 8位图像最大值就是255
    return rst

def threshold1(save_path,img_path,type=".png"):
    if os.path.exists(save_path):                               # 判断文件是否存在
        pass
    else:
        os.makedirs(save_path)                                  # 创建文件
    str=img_path+'/*'+ type
    coll = io.ImageCollection(str,load_func=convert)            # 批处理
    for i in range(len(coll)):
        io.imsave(save_path + np.str(i) + '.png',coll[i])       # 保存图片在