import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class Conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)


class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1 = Conv(3,64)
        self.conv2 = Conv(64,64)
        self.conv3 = Conv(64,128)
        self.conv4 = Conv(128,128)
        self.conv5 = Conv(128,256)
        self.conv6 = Conv(256,256)
        self.conv7 = Conv(256,256)
        self.conv8 = Conv(256,512)
        self.conv9 = Conv(512,512)
        self.conv10 = Conv(512,512)
        self.conv11 = Conv(512,512)
        self.conv12 = Conv(512,512)
        self.conv13 = Conv(512,512)
        self.MaxPool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.MaxPool(x1)

        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x2 = self.MaxPool(x2)

        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x3 = self.MaxPool(x3)

        x4 = self.conv8(x3)
        x4 = self.conv9(x4)
        x4 = self.conv10(x4)
        x4 = self.MaxPool(x4)

        x5 = self.conv11(x4)
        x5 = self.conv12(x5)
        x5 = self.conv13(x5)
        x5 = self.MaxPool(x5)

        return x1,x2,x3,x4,x5


class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.VGG = VGG()
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

         
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        features = self.VGG(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)))

        return y

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = FCN().to(device)

    summary(vgg, (3, 224, 224))
