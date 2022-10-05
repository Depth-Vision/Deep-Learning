import os
import csv
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
