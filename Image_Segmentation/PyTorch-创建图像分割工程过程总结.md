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

```apache
import torch
import torch.nn as nn
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self,ch_in=3,ch_out=64):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out = self.conv(x)
        return out

class up_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_block,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out = self.up(x)
        return out

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(U_Net,self).__init__()
        self.Maxpoool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.conv2 = conv_block(ch_in=32,ch_out=64)
        self.conv3 = conv_block(ch_in=64,ch_out=128)
        self.conv4 = conv_block(ch_in=128,ch_out=256)
        self.conv5 = conv_block(ch_in=256,ch_out=512)

        self.up5 = up_block(ch_in=512,ch_out=256)
        self.up_conv5 = conv_block(ch_in=512,ch_out=256)

        self.up4 = up_block(ch_in=256,ch_out=128)
        self.up_conv4 = conv_block(ch_in=256,ch_out=128)

        self.up3 = up_block(ch_in=128,ch_out=64)
        self.up_conv3 = conv_block(ch_in=128,ch_out=64)
      
        self.up2 = up_block(ch_in=64,ch_out=32)
        self.up_conv2 = conv_block(ch_in=64,ch_out=32)

        self.Conv1_1 = nn.Conv2d(in_channels=32,out_channels=output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.Maxpoool(x1)
        x2 = self.conv2(x2)
        x3 = self.Maxpoool(x2)
        x3 = self.conv3(x3)
        x4 = self.Maxpoool(x3)
        x4 = self.conv4(x4)
        x5 = self.Maxpoool(x4)
        x5 = self.conv5(x5)

        d5 = self.up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.Conv1_1(d2)
        d1 = torch.sigmoid(d1)

        return d1

if __name__ == "__main__":
    unet = U_Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = U_Net().to(device)
    summary(unet,(3,224,224))
```
