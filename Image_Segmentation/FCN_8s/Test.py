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

    
    
