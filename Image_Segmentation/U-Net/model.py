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