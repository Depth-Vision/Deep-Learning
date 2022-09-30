import os 
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from  torchvision import transforms
from torch.utils.data import Dataset

class DataInput(Dataset):
    def __init__(self,img_h,img_w,path,data_file,label_file,preprocess):
        """
        数据初始化
        img_h: resize图像高度
        img_w: resize图像宽度
        path: 数据集路径
        data_file: 原始数据文件名
        label_file: 数据标签文件名
        preprocess: 是否进行数据预处理
        """
        self.img_h = img_h
        self.img_w = img_w
        self.path = path
        self.data_file = data_file
        self.label_dile = label_file
        self. preprocess = preprocess
        self.file_list = os.listdir(path+"/"+data_file)
        print("Data Loading.....................")
        print("DataPath: " + path , "     DataSize: " + str(len(self.file_list)))
        print("————————————————————————————————————————————")

    def __len__(self):
        return len(self.file_list)

            
