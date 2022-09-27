import os
import time
import torch 
import DataInput
from model import LeNet
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


       




                    
                    



















