# Pytorch 图像分割工程过程总结（U-Net为例）

1.数据集准备

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

```python
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

```python
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

## 4.构建评测标准计算文件

ISI性能指标：

<center><b><font size=4>像素点分类</font></b></center>

| 预测值 | 标签为1 | 标签为0 |
| :----: | :-----: | :-----: |
|   1   |   TP   |   FP   |
|   0   |   FN   |   TN   |

**IOU交并比：**


$$
IOU=\frac{N_{TP}}{N_{TP}+N_{FP}+N_{FN}}
$$

**ACC准确度：**

    

$$
ACC=\frac{N_{TP}+N_{TN}}{N_{TP}+N_{TN}+N_{FP}+N_{FN}}
$$

**Precision查准率：**

    

$$
Precision=\frac{N_{TP}}{N_{TP}+N_{FP}}
$$

**Recall召回率：**

    

$$
Recall=\frac{N_{TP}}{N_{TP}+N_{FN}}
$$

**根据以上公式建立指标计算函数(ISI.py)**:

```python
import torch
import numpy as np

def DataInput(Pre_Label,Tre_Label):
    num0, dim0, pixel00, pixel01 = Pre_Label.shape
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    if(Pre_Label.shape == Tre_Label.shape):
        for i in range(num0):
            for j in range(dim0):
                for k in range(pixel00):
                    for l in  range(pixel01):
                        pre_value = Pre_Label[i][j][k][l]
                        tre_value = Tre_Label[i][j][k][l]
                        if(pre_value == tre_value == 1):
                            TP += 1
                          
                        if(pre_value == 1 and tre_value == 0):
                            FP += 1
                          
                        if(pre_value == 0 and tre_value == 1):
                            FN += 1
                          
                        if(pre_value == tre_value == 0):
                            TN += 1
                          
    else:
        print("Size error:")
        print("Pre_Label size(",Pre_Label.shape,")!=Tre_Label size(",Tre_Label.shape,")")
    if TP == 0:
        print("TP=0")
    if FP == 0:
        print("FP=0")
    if FN == 0:
        print("FN=0")
    if TN == 0:
        print("TN=0")
    return TP,FP,FN,TN

def mIOU(Pre_Label,Tre_Label):
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    IOU = TP/(TP+FP+FN)
    return IOU

def mACC(Pre_Label,Tre_Label):
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    AP = (TP + TN)/(TP+TN+FP+FN)
    return AP
      
def mPrecision(Pre_Label,Tre_Label):
    TP,FP,FN,TN, = DataInput(Pre_Label,Tre_Label)
    Precision = TP/(TP + FP)
    return Precision

def mRecall(Pre_Label,Tre_Label):
    TP,FP,FN,TN, = DataInput(Pre_Label,Tre_Label)
    Recall = TP/(TP + FN)
    return Recall

def ISI(epochs,Pre_Label,Tre_Label):
    """
    Image segmentation index:图像分割指标
    IOU:交并比
    ACC:平均准确度
    Precision:查准率
    Recall:召回率
    """
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    out = {"epoch":str(epochs)}
    return[str(epochs),str(round(TP/(TP+FP+FN),4)),str(round((TP + TN)/(TP+TN+FP+FN),4)),str(round(TP/(TP+FP),4)),str(round(TP/(TP+FN),4))]
```

## 5.构建训练文件

构建模型训练文件Train.py

```python
import os
import csv
import ISI
import Mat
import torch
import numpy as np
from model import U_Net
from torch.utils import data
from DataInput import DataInput
from torchvision.utils import save_image

class Trainer(object):
    def __init__(self,lr=0.5,batch_size=16,num_epoch=60,train_set=None,model_path="./RunRecord/model/",device="cpu"):
        """
        lr: 学习率
        batch_size: 批量大小
        num_epoch: 迭代周期数
        train_set: 训练数据集
        model_path: 模型保存路径
        device: 训练设备
        """
        self.lr = lr
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_path = model_path

        # 初始化数据载入
        self.data_loader = data.DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True,num_workers=0)

        # 初始化模型
        self.Net = U_Net()
        if device.lower() == "cpu":
            self.device = torch.device("cpu")
            self.device1 = "CPU"
        else:
            self.device = torch.device("cuda")
            self.device1 = "GPU"
        self.Net.to(self.device)

        # 初始化损失函数
        self.loss = torch.nn.MSELoss()

        # 初始化学习率优化函数（学习率迭代优化器）
        self.optim = torch.optim.Adam(self.Net.parameters(),lr=self.lr,betas=(0.5,0.999))

    # 模型训练
    def train(self):
        print("Model: AAMC_Net " + "      Device: " + self.device1)
        print("BatchSize: " + str(self.batch_size) + "          Epoch: " + str(self.num_eopch))
        print("————————————————————————————————————————————")
        best_loss = 1000000
        num = 0

        # 批循环
        for epoch in range(self.num_epoch):
            self.Net.train(True)
            epoch_loss = 0
            print("Epoch:",epoch+1,"/",self.num_eopch,"      Training.............")
            epoch1 = epoch

            # 迭代循环
            for i,(bx,by) in enumerate(self.data_loader):
                num += 1
                bx = bx.to(self.device)
                by = by.to(self.device)
                bx_gen = self.Net(bx)
                loss = self.loss(bx_gen,by)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                epoch_loss +=loss.item()
            print("loss: ",epoch_loss)

            # 保存最优损失率的模型
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                if os.path.exists(self.model_path) is False:
                    os.makedirs(self.model_path)
                torch.save(self.Net.state_dict(),self.model_path + "/U_Net.plk")
        print("————————————————————————————————————————————")
        print("Training is over")
        print("Model saved in: " + self.model_path)
        print("Training log in: " + "./RunRecord")   

    # 训练效果图像及评价指标输出
    def save_img(self,epochs,save_path="./RunRecord/Test_Images",save_name="result"):
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        data_iter = iter(self.data_loader)
        img,labels = next(data_iter)
        self.Net.eval()

        # 设置网络无梯度
        with torch.no_grad():
            bx_gen = self.Net(img.to(self.device))

        # CPU下训练结果概率过滤
        if (self.device1 == "CPU"):
            img = img.data.cpu()[:5]
            gen_label = bx_gen.data.cpu()[:5]
            labels = labels.data.cpu()[:5]
            gen_label0 = torch.where(gen_label>0.5,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label1 = torch.where(gen_label>0.1,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label2 = torch.where(gen_label>0.01,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label3 = torch.where(gen_label>0.001,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label4 = torch.where(gen_label>0.0001,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            labels = torch.where(labels>0.5,torch.full_like(gen_label,0),torch.full_like(gen_label,1)).cpu()
            labels = torch.zeros([3,224,224]).cpu() + labels

        # GPU下训练结果概率过滤
        if (self.device1 == "GPU"):
            img = img.data.cuda()[:5]
            gen_label = bx_gen.data.cuda()[:5]
            labels = labels.data.cuda()[:5]
            gen_label0 = torch.where(gen_label>0.5,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label1 = torch.where(gen_label>0.1,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label2 = torch.where(gen_label>0.01,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label3 = torch.where(gen_label>0.001,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            gen_label4 = torch.where(gen_label>0.0001,torch.full_like(gen_label,1),torch.full_like(gen_label,0))
            labels = torch.where(labels>0.5,torch.full_like(gen_label,0),torch.full_like(gen_label,1)).cpu()
            labels = torch.zeros([3,224,224]).cuda() + labels
  
        # 评测指标计算及保存
        out = open("./RunRecord/ISI_Record.csv","a+")
        csv_writer = csv.writer(out, dialect = "excel",lineterminator = '\n')
        key = ["Epoch","IOU","ACC","Precision","Recall"]
        a = ISI.ISI(epochs+1,gen_label2,labels)
        if(epochs == 0):
            csv_writer.writerow(key)
            csv_writer.writerow(a)
        else:
            csv_writer.writerow(a)

        # 训练效果图像保持
        save_tensor = torch.cat([img,gen_label0,gen_label1,gen_label2,gen_label3,gen_label4,labels],0)            
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        save_image(save_tensor,save_path+'/'+save_name,nrow=5)


if __name__ == '__main__':
    # 读取数据
    torch.cuda.empty_cache()
    train_data = DataInput(img_w=224,img_h=224,path='./DATA',data_file='images',label_files="profiles",preprocess=True)
    # 构建模型，训练模型
    trainer = Trainer(lr=0.000001,batch_size=2,num_epoch=16,train_set=train_data,device="gpu")
    trainer.train()
    # 评测指标数据可视化
    Mat.ISI_Chart()
    Mat.Ite_Loss_Chart()

```

## 6.构建数据可视化文件

使用matplotlib使在模型训练过程中计算出的模型训练测评指标可视化

构建Mat.py文件

```python
import csv
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))      # 用来正常显示中文标签
      
def ISI_Chart():
    path="./RunRecord/ISI_Record.csv"
    title="ISI_Chart"

    plt.rcParams['font.sans-serif']=['SimHei']                              # 用来正常显示负号 
    plt.rcParams['axes.unicode_minus']=False

    with open(path) as f:                                                   # 打开文件文件并将内容储存在reader中
        reader=csv.reader(f)                                                # 读取并将内容储存在列表reader中
        next(reader)                                                        # next()函数获取第一行，即文件头
        Epoch,IOU,ACC,Precision,Recall = [],[],[],[],[]
      
        for row in reader:
            epoche = float(row[0])
            Epoch.append(epoche)

            iou = float(row[1])
            IOU.append(iou)

            ac = float(row[2])
            ACC.append(ac)

            pre = float(row[3])
            Precision.append(pre)

            rec = float(row[4])
            Recall.append(rec)


    plt.plot(Epoch,IOU,c="red",label="IOU",linewidth=1)
    plt.plot(Epoch,ACC,c="blue",label="ACC",linewidth=1)
    plt.plot(Epoch,Precision,c="green",label="Precision",linewidth=1)
    plt.plot(Epoch,Recall,c="black",label="Recall",linewidth=1,linestyle='--')
    plt.xlabel('epoch') 
    plt.ylabel('ratio')
    plt.title(title) 
    plt.legend() 
    plt.savefig('./RunRecord/ISI_Chart.svg')
    plt.savefig('./RunRecord/ISI_Chart.png')
    plt.close()

if __name__ == '__main__':
    ISI_Chart()
```

## 7.构建模型测试文件

构建Test.py文件进行效果模型测试

```python
import os
import time
import torch
import shutil
from PIL import Image
from model import AAMC_Net
from gray import Gray,threshold1
from torchvision import transforms
from torchvision.utils import save_image


device = torch.device('cuda')                                                                   # 配置设备
image = transforms.Compose([                                                                    # 图片处理
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),                                       # 如出现RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]报错则调整此句有无
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
label = transforms.Compose([                                                                    # 标签处理
                transforms.ToTensor(),
                ])


def prediect(a,img_path,save_path):                                                             # 预测函数
    model = AAMC_Net().cuda()                                                                   # 加载模型
    model.load_state_dict(torch.load("./AAMC_Net.pkl"))
    net=model.to(device)
    torch.no_grad()                                                                             # 张量计算无需计算梯度
    img=Image.open(img_path)
    img=image(img).unsqueeze(0)                                                                 # 数据维度扩充
    img_ = img.to(device)
    outputs = net(img_)*2
    result = torch.where(outputs>0,torch.full_like(outputs,1),torch.full_like(outputs,0))       # 预测结果阈值处理
    result = result.cpu()
    name = str(a)+".png"
    save_image(result,save_path+'/'+name,nrow=1)                                                # 保存文件


def Verification(img_path,save_path):
    if os.path.exists("./DATA/test/gray"):                                                      # 判断灰度文件是否存在
        pass
    else:
        os.makedirs("./DATA/test/gray")                                                         # 创建灰度文件
    Gray(save_path="./DATA/test/gray/",img_path=img_path,type=".jpg")                           # 图像批量灰度化
    img_path = "./DATA/test/gray"
    if os.path.exists("./DATA/test/temporary"):                                                 # 判断缓存文件是否存在
        pass
    else:
        os.makedirs("./DATA/test/temporary")                                                    # 创建缓存文件
    file_list = os.listdir(img_path)                                                          
    num = len(file_list)
    time_start = time.time()                                                                    # 开始计时
    for item in range(len(file_list)):
        img_name = file_list[item]
        img_path1 = img_path+"/"+img_name
        prediect(item,img_path1,save_path)
    time_end = time.time()                                                                      # 结束计时
    time_c= time_end - time_start                                                               # 运行所花时间
    timea = time_c/num
    print('time cost', timea, 's')
    Gray(save_path="./DATA/test/temporary/",img_path=save_path)
    threshold1(save_path=save_path,img_path="./DATA/test/temporary/")
    shutil.rmtree("./DATA/test/gray")                                                           # 清理灰度文件
    shutil.rmtree("./DATA/test/temporary")                                                      # 清理缓存文件
  

     
      

if __name__ == '__main__':
    torch.cuda.empty_cache()                                                                    #清理显存
    Verification("./DATA/test/img","./DATA/test_result/")                                       #（图片路径，预测结果保存路径）

  
  

```
