from xdrlib import ConversionError
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                )
    def forward(self,x):
        out = self.conv(x)
        return out

class VGG_Net(nn.Module):
    def __init__(self,classes_num):
        super(VGG_Net,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv_1 = Conv_block(3,64)
        self.Conv_2 = Conv_block(64,64)

        self.Conv_3 = Conv_block(64,128)
        self.Conv_4 = Conv_block(128,128)

        self.Conv_5 = Conv_block(128,256)
        self.Conv_6 = Conv_block(256,256)
        self.Conv_7 = Conv_block(256,256)

        self.Conv_8 = Conv_block(256,512)
        self.Conv_9 = Conv_block(512,512)
        self.Conv_10 = Conv_block(512,512)

        self.Conv_11 = Conv_block(512,512)
        self.Conv_12 = Conv_block(512,512)
        self.Conv_13 = Conv_block(512,512)
        
        self.fc_layers = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, classes_num),
                )
    def forward(self,x):
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Maxpool(x)

        x = self.Conv_3(x)
        x = self.Conv_4(x)
        x = self.Maxpool(x)

        x = self.Conv_5(x)
        x = self.Conv_6(x)
        x = self.Conv_7(x)
        x = self.Maxpool(x)

        x = self.Conv_8(x)
        x = self.Conv_9(x)
        x = self.Conv_10(x)
        x = self.Maxpool(x)
        
        x = self.Conv_11(x)
        x = self.Conv_12(x)
        x = self.Conv_13(x)
        x = self.Maxpool(x)

        x = x.view(-1,512*7*7)

        x = self.fc_layers(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VGG_Net_ = VGG_Net(10).to(device)
    summary(VGG_Net_,input_size=(3,224,224))
