# Pytorch 图像分割工程过程总结（U-Net为例）

## 1.数据集准备

数据集DATA文件夹下包含images、labels文件夹，images文件里存放原始数据图片，labels文件里存放标签数据图片，两文件里的图像名一一对应

```apache
|-Data
	|-images
		|-img1.jpg
		|-img2.jpg
	|-labels
		|-img1.png
		|-img2.png
```

## 2.数据预处理

创建DataInput.py文件，进行数据读取、预处理、装载

```apache
from asyncio import transports
import os 
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from  torchvision import transforms
from torch.utils.data import Dataset

class DataInput(Dataset):
    def __init__(self,img_h,img_w,path="../DATA",data_file="images",label_file="labels",preprocess=True):
        """
        数据初始化
        img_h: resize图像高度
        img_w: resize图像宽度
        path: 数据集路径
        data_file: 原始数据文件名
        label_file: 数据标签文件名
        preprocess: 是否进行数据预处理
        """
        self.file_list = os.listdir(path+"/"+data_file)
        self.img_h = img_h
        self.img_w = img_w
        self.path = path
        self.data_file = data_file
        self.label_file = label_file
        self. preprocess = preprocess
  
        print("Data Loading.....................")
        print("DataPath: " + path , "     DataSize: " + str(len(self.file_list)))
        print("————————————————————————————————————————————")

    def __len__(self):
        # 返回数据集大小
        return len(self.file_list)                                                                

    def __getitem__(self, item):
        # 索引数据集
        img_name = self.file_list[item]
        label_name = img_name.split(".")[0]
        img_path = self.path + "/" + self.data_file + "/" + img_name
        label_path = self.path + "/" + self.label_file + "/" + "/" + label_name + ".png"

        # 读取数据
        img = Image.open(img_path)
        label = Image.open(label_path)

        # 数据预处理
        if self.preprocess:
            trans_img = transforms.Compose([
                transforms.Resize(size=(self.img_w,self.img_h)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            img = trans_img(img)
            trans_label = transforms.Compose([
                transforms.Resize(size=(self.img_w,self.img_h)),
                transforms.ToTensor()
            ])
            label = trans_label(label)

        return img,label

if __name__ == "__main__":
    trans_data = DataInput(img_h=224,img_w=224)
    img,label = trans_data.__getitem__(5)
    print(img.size())
    print(label.size())
    plt.imshow(img.data.numpy().transpose([1,2,0]))
    plt.show()
    plt.imshow(label.data.numpy().transpose([1,2,0]))
    plt.show()
      

```


## 3.模型搭建

创建model.py文件，模型手动搭建或引用，以U-Net为例。

```

```
