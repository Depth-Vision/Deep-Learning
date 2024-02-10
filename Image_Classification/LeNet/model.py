import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet(nn.Module):            
    def __init__(self,classes_num):                
        super(LeNet, self).__init__()  
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)
                           
    def forward(self, x):            # input(1, 32, 32)  
        x = F.relu(self.conv1(x))    # output(6, 28, 28)
        x = self.pool1(x)            # output(6, 14, 14)
        x = F.relu(self.conv2(x))    # output(16, 10, 10)
        x = self.pool2(x)            # output(16, 5, 5)
        x = x.view(-1, 16*5*5)       # output(16*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(classes_num)
        return x

if __name__ == "__main__":

    # print(LeNet(15))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LeNet_ = LeNet(15).to(device)
    summary(LeNet_, input_size=(1, 32, 32))
