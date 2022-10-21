
import os
import torch 
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet
import cv2 as cv

def test(path,model_path):
    """
    path: 测试集地址
    model: 网络权重参数文件地址
    """
    # 打开类别文本文件
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