import torch.nn as nn
import torch.nn.functional as F
class convNet_val(nn.Module):
    def __init__(self,in_channels):
        super(convNet_val,self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        #                dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=64,momentum=0.1)
        self.c2 = nn.Conv2d(64,64,5,1,2)
        self.bn2 = nn.BatchNorm2d(num_features=64,momentum=0.1)
        self.m1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(0.2)
        
        self.c3 = nn.Conv2d(64,128,5,1,2)
        self.bn3 = nn.BatchNorm2d(128,0.1)
        self.c4 = nn.Conv2d(128,128,5,1,2)
        self.bn4 = nn.BatchNorm2d(128,0.1)
        self.m2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(0.2)
        
        self.c5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256,0.1)
        self.c6 = nn.Conv2d(256,256,3,1,1)
        self.bn6 = nn.BatchNorm2d(256,0.1)
        self.m3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256*3*3,256)
        self.d4 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256,128)
        self.d5 = nn.Dropout(0.1)
        self.out = nn.Linear(128,10)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.c1(x)),negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.c2(x)),0.1)
        x = self.m1(x)
        x = self.d1(x)
        
        x = F.leaky_relu(self.bn3(self.c3(x)),0.1)
        x = F.leaky_relu(self.bn4(self.c4(x)),0.1)
        x = self.m2(x)
        x = self.d2(x)
        
        x = F.leaky_relu(self.bn5(self.c5(x)),0.1)
        x = F.leaky_relu(self.bn6(self.c6(x)),0.1)
        x = self.m3(x)
        x = self.d3(x)
        
        x = x.view(-1, 256*3*3) #reshape
        x = F.leaky_relu(self.fc1(x),0.1)
        x = self.d4(x)
        x = F.leaky_relu(self.fc2(x),0.1)
        x = self.d5(x)
        return self.out(x)



class convNet(nn.Module):
    def __init__(self,in_channels):
        super(convNet,self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        #                dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=64,momentum=0.1)
        self.c2 = nn.Conv2d(64,64,5,1,2)
        self.bn2 = nn.BatchNorm2d(num_features=64,momentum=0.1)
        self.m1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(0.2)
        
        self.c3 = nn.Conv2d(64,128,5,1,2)
        self.bn3 = nn.BatchNorm2d(128,0.1)
        self.c4 = nn.Conv2d(128,128,5,1,2)
        self.bn4 = nn.BatchNorm2d(128,0.1)
        self.m2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(0.2)
        
        self.c5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256,0.1)
        self.c6 = nn.Conv2d(256,256,3,1,1)
        self.bn6 = nn.BatchNorm2d(256,0.1)
        self.m3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256*3*3,256)
        self.out = nn.Linear(256,10)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.c1(x)),negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.c2(x)),0.1)
        x = self.m1(x)
        x = self.d1(x)
        
        x = F.leaky_relu(self.bn3(self.c3(x)),0.1)
        x = F.leaky_relu(self.bn4(self.c4(x)),0.1)
        x = self.m2(x)
        x = self.d2(x)
        
        x = F.leaky_relu(self.bn5(self.c5(x)),0.1)
        x = F.leaky_relu(self.bn6(self.c6(x)),0.1)
        x = self.m3(x)
        x = self.d3(x)
        
        x = x.view(-1, 256*3*3) #reshape
        x = F.leaky_relu(self.fc1(x),0.1)
        return self.out(x)