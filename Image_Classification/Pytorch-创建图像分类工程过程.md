# PyTorch 图像分类工程过程总结(LeNet为例)

## 1.数据集准备

数据集Data文件夹下包含train、valid文件夹，train、valid文件夹数据按照各自类别放入到不同类别文件夹

例如：

```apache
|-Data
	|-train
		|-Class1
		|-Class2
	|-valid
		|-Class1
		|-Class2
```

## 2.数据预处理

创建DataInput.py文件，进行数据读取、预处理、装载

```python
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class DataInput():
    def __init__(self,img_w=32,img_h=32,batch_size=4,data_path=None):
        """
        数据预处理初始化:
        img_w: 模型输入图像宽
        img_h: 模型输入图像高
        batch_size: 批量大小
        datapath: 数据存放地址
        """

        self.img_size = [img_w,img_h]
        self.batch_size = batch_size
        self.data_path = data_path
        self.datasets = 0

        print("Data Loading " + "."*10)
        print("DataPath: ",self.data_path)

    def data_size(self,phase=None):
        return len(self.datasets[phase])

    def classes_num(self):
        classes_num = len(os.listdir(self.data_path + "/train"))
        return classes_num

    def preprocess(self):
        data_path = self.data_path
        # 数据预处理
        data_transform = {x:transforms.Compose([transforms.Resize(self.img_size),
                                       transforms.ToTensor()]) for x in ['train', 'valid']}   
        # 读取数据
        image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_path,x),
                                        transform = data_transform[x]) for x in ['train', 'valid']}  
        # 数据装载
        dataloader = {x:DataLoader(dataset = image_datasets[x],
                                           batch_size = self.batch_size,
                                           shuffle = True) for x in ['train', 'valid']} 
        self.datasets = image_datasets 
        return dataloader  
  
```

## 3.搭建模型

创建model.py文件。模型随意搭建或引用。以LeNet为例。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet(nn.Module):      
    def __init__(self,classes_num):          
        super(LeNet, self).__init__()  
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)
                     
    def forward(self, x):            # input(3, 32, 32)  
        x = F.relu(self.conv1(x))    #output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(classes_num)
        return x

if __name__ == "__main__":

    # print(LeNet(15))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LeNet_ = LeNet(15).to(device)
    summary(LeNet_, input_size=(3, 32, 32))
  
```

## 4.模型训练

创建Train.py文件

###### 1.初始化数据（变量、模型、学习率优化器、损失函数等）。

###### 2.训练模型（加载数据、数据输入网络、网络权重参数反向传递、损失计算、准确率计算、模型保存）。

**注：要同时考虑训练集和测试集。**

```python
import os
import time
import torch 
import DataInput
from model1 import LeNet
from torch.autograd import Variable

class Trainer():
    def __init__(self,lr,batch_size,num_epoch,img_w,img_h,data_path,model_path,device):
        """
        模型训练初始化:
        lr: 初始学习率
        batch_size: 批量大小
        num_epoch: 迭代周期数
        img_w: 模型输入图像宽
        img_h: 模型输入图像高
        data_path: 数据存放地址
        model_path: 模型保存地址
        device: 训练设备(CPU or GPU)
        """
        self.lr = lr 
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.img_w = img_w
        self.img_h = img_h
        self.data_path = data_path
        self.model_path = model_path
        self.data = DataInput.DataInput(self.img_w,self.img_h,self.batch_size,self.data_path)
        self.loss = torch.nn.CrossEntropyLoss()
        self.classes_num = self.data.classes_num()
        # 模型初始化
        self.model = LeNet(self.classes_num)
        if device == "gpu" or "GPU":
            self.device = torch.device('cuda')
            self.device_name = "GPU"
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            self.device_name = "CPU"
        self.model.to(self.device)   
        # 学习率初始化及学习率优化器   
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.lr,betas=(0.5,0.999))  

    # 保存模型网络及权重参数
    def save_all_model(self,mode):

        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        torch.save(self.model,self.model_path + "/"+str(mode) + "_model.pth")

    # 仅保存模型网络权重参数
    def save_model_weight(self,mode):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(),self.model_path + "/"+str(mode) + "_model_weight.pth")

    # 模型训练
    def train(self):
        best_loss = 10000000
        # 加载数据
        dataloader = self.data.preprocess()
        print("Training "+"."*10)
        print("BatchSize: " + str(self.batch_size))
        print("Classes_num: ",self.classes_num)
        print("Epoch: " + str(self.num_epoch))
        print("Device: " + self.device_name)
        print("-" * 20)

        # 训练开始
        time_open = time.time()
        for epoch in range(self.num_epoch):
            print("Epoch {}/{}".format(epoch + 1,self.num_epoch))
            print("-" * 20)
            # 判断输入数据时训练集还是测试集来确定是否开启网络权重参数传递
            for phase in ["train","valid"]:
                if phase == "train":
                    print("train......")
                    self.model.train(True)
                else:
                    print("valid......")
                    self.model.train(False)        
                run_loss = 0.0
                num_correct = 0.0
                # 从数据装载器里抽出数据输入网络进行训练
                for batch,(image,label) in enumerate(dataloader[phase],1):
                    image = Variable(image).to(self.device)
                    label = Variable(label).to(self.device)
                    pred_label = self.model(image)
                    _ , pred = torch.max(pred_label.data,1)
                    loss = self.loss(pred_label,label)
                    self.optim.zero_grad()
                    # 判断输入数据是否为训练集，来确认是否进行反向传播和梯度更新
                    if phase == "train":
                        loss.backward()
                        self.optim.step()
                    run_loss += loss.item()
                    num_correct += torch.sum(pred == label)
                    # 每15个批次输出一次训练loss和Acc
                    if batch % 15 == 0 and phase == "train":
                        print("batch:{}   trainloss:{:.6f}   trainACC:{:.4f}%".format(batch,run_loss / batch,
                                                                               100 * num_correct / (self.batch_size * batch)))
                epoch_loss_v = run_loss * self.batch_size / self.data.data_size(phase=phase)
                epoch_acc = num_correct / self.data.data_size(phase=phase)
                print("{}   Loss: {:.6f}   Acc: {:.4f} %".format(phase,epoch_loss_v,epoch_acc * 100))

                # 比较损失，保存最优模型
                if run_loss < best_loss:
                    best_loss = run_loss
                    self.save_all_model(mode="best")
                    self.save_model_weight(mode="best")
                # 保存最后模型
                self.save_all_model(mode="last")
                self.save_model_weight(mode="last")
        time_end = time.time()   
        train_time = (time_end - time_open) / 60
      
        print("-"*20)
        print("Training is over")
        print("Training time: " + str(train_time) + "min")
        print("Model saved in: " + self.model_path)



if __name__ == "__main__":
    Train = Trainer(lr=0.001,batch_size=8,num_epoch=20,img_w=32,img_h=32,data_path="./Data",model_path="./model",device="GPU")
    Train.train()

```

## 5.模型测试

创建Classes.txt文件，文件中每行一个类别标签，与训练集标签顺序对应。

创建test.py 文件,测试模型对各个类别识别的准确度。

```python
import os
import torch 
import torchvision.transforms as transforms
from PIL import Image
from model1 import LeNet
import cv2 as cv

def test(path,model_path):
    """
    path: 测试集地址
    model: 网络权重参数文件地址
    """
    # classes = ("B1","B2","B3","B4","B5","BB","BS","NO","R1","R2","R3","R4","R5","RB","RS")
    C = open("Classes.txt",encoding="gbk")
    classes = []
    for line in C:
        classes.append(line.strip())
    classes = tuple(classes)
    # print(classes)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    net = LeNet(len(classes))
    net.load_state_dict(torch.load(model_path))
    file_list = os.listdir(path)
    cla_prve = file_list[0]
    for cla in file_list:
        print("-"*10)
        print("Classes:",cla)
        img_path = path + "/" + cla
        # print(img_path)
        img_list = os.listdir(img_path)
        pre_correct = 0
        if cla == cla_prve:
            pass
        else:
            pre_correct = 0
        for item in range(len(img_list)):
            img_path_n = img_path + "/" + img_list[item] 
            img = Image.open(img_path_n).convert("RGB")
            img = transform(img)
            img = torch.unsqueeze(img,dim=0)
            with torch.no_grad():
                Y = net(img)
                predict = torch.max(Y,1)
                predict = predict[1].numpy()
                if classes[int(predict)] == cla:
                    pre_correct += 1
                # else:
                #     print(img_path_n,"Predict error, Label: " + cla + "  predict: " + classes[int(predict)])
        print("Classes: " + cla + "    ACC: ",str((pre_correct/len(img_list))*100) + "%")
  

if __name__ == "__main__":
    test("./Data/valid","./model/last_model_weight.pth")
  
```

。
