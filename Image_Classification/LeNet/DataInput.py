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
        data_transform = {x:transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()]) for x in ['train', 'valid']}   
        # 读取数据
        image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_path,x),transform = data_transform[x]) for x in ['train', 'valid']}  
        # 数据装载
        dataloader = {x:DataLoader(dataset = image_datasets[x],batch_size = self.batch_size,shuffle = True) for x in ['train', 'valid']} 
        self.datasets = image_datasets 
        return dataloader    


