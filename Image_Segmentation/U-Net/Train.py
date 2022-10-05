from cProfile import label
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
        self.data_loader = data.DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True,num_workers=0)

        self.Net = U_Net()
        if device.lower() == "cpu":
            self.device = torch.device("cpu")
            self.device1 = "CPU"
        else:
            self.device = torch.device("cuda")
            self.device1 = "GPU"
        
        self.Net.to(self.device)
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.Net.parameters(),lr=self.lr,betas=(0.5,0.999))

    def train(self):
        print("Model: AAMC_Net " + "      Device: " + self.device1)
        print("BatchSize: " + str(self.batch_size) + "          Epoch: " + str(self.num_eopch))
        print("————————————————————————————————————————————")
        best_loss = 1000000
        num = 0

        for epoch in range(self.num_epoch):
            self.Net.train(True)
            epoch_loss = 0
            print("Epoch:",epoch+1,"/",self.num_eopch,"      Training.............")
            epoch1 = epoch

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

            if best_loss > epoch_loss:
                best_loss = epoch_loss
                if os.path.exists(self.model_path) is False:
                    os.makedirs(self.model_path)
                torch.save(self.Net.state_dict(),self.model_path + "/U_Net.plk")
        print("————————————————————————————————————————————")
        print("Training is over")
        print("Model saved in: " + self.model_path)
        print("Training log in: " + "./RunRecord")     

    def save_img(self,epochs,save_path="./RunRecord/Test_Images",save_name="result"):
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        data_iter = iter(self.data_loader)
        img,labels = next(data_iter)
        self.Net.eval()

        with torch.no_grad():
            bx_gen = self.Net(img.to(self.device))
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

        out = open("./RunRecord/ISI_Record.csv","a+")
        csv_writer = csv.writer(out, dialect = "excel",lineterminator = '\n')
        key = ["Epoch","IOU","ACC","Precision","Recall"]
        a = ISI.ISI(epochs+1,gen_label2,labels)
        if(epochs == 0):
            csv_writer.writerow(key)
            csv_writer.writerow(a)
        else:
            csv_writer.writerow(a)

        save_tensor = torch.cat([img,gen_label0,gen_label1,gen_label2,gen_label3,gen_label4,labels],0)                  # 图片结果保存
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
    Mat.ISI_Chart()
    Mat.Ite_Loss_Chart()
