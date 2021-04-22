import torch
import torch.nn as nn

class TwoView_Classification_Model(nn.Module): 
    def __init__(self, class_num=2, dropout=0.5, return_feature=False, add_bias=False):
        super(TwoView_Classification_Model, self).__init__()
        self.feature_extractor1 = S3D_G(return_feature=True) #renturn (:, 1024)
        self.feature_extractor2 = S3D_G(return_feature=True)
        self.feature_length = 1024    
        self.class_num = class_num
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_length*2, class_num, bias=add_bias)
        self.return_feature = return_feature

    def baseline(self,x1,x2):
        f1 = self.feature_extractor1(x1)
        f2 = self.feature_extractor2(x2)
        x = torch.cat([f1, f2], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        if self.return_feature:
            return x, f1, f2
        else:
            return x
                     
    def forward(self, x1, x2):
        return self.baseline(x1,x2)


class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm3d(out_channel,
                                 eps=0.001, 
                                )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class S3D_G_block(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(S3D_G_block, self).__init__()

        self.branch1 = BasicConv3d(in_channel,out_channel[0], kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[1], kernel_size=1, stride=1),
            BasicConv3d(out_channel[1], out_channel[1],kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(out_channel[1], out_channel[2], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[3], kernel_size=1, stride=1),
            BasicConv3d(out_channel[3], out_channel[3], kernel_size=(1, 3, 3), stride=1, padding= (0, 1, 1)),
            BasicConv3d(out_channel[3], out_channel[4], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3,stride=1,padding=1),
            BasicConv3d(in_channel, out_channel[5], kernel_size=(3,1,1), stride=1,padding=(1,0,0))
        )
        self.squeeze = nn.AdaptiveAvgPool3d(1)

        self.excitation = nn.Conv1d(1, 1, (3,1,1), stride=1,padding=(1,0,0))
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x=torch.cat([x1,x2,x3,x4], 1)
        input = x
        x=self.squeeze(x)
        x=self.excitation(x.permute(0,2,1,3,4))
        x=self.sigmoid(x)
        return x.permute(0,2,1,3,4)*input


class S3D_G(nn.Module):
    # Input size: 64x224x224
    def __init__(self, num_class=2, return_feature=False):
        super(S3D_G, self).__init__()
        self.return_feature = return_feature
        self.conv1=BasicConv3d(3,64,kernel_size=7,stride=2,padding=3) # DHW = 32,112,112
        self.pool1=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)) #32, 56,56
        self.conv2=BasicConv3d(64,64,kernel_size=1,stride=1) #32，56，56
        self.conv3=BasicConv3d(64,192,kernel_size=3,stride=1,padding=1) #32，56，56
        self.pool2=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)) #32， 28，28
        self.Inception1=nn.Sequential(S3D_G_block(192, [64,96,128,16,32,32]),
                                      S3D_G_block(256, [128, 128, 192, 32, 96, 64])) #32，28，28
        self.pool3=nn.MaxPool3d(kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1)) #16，14，14
        self.Inception2=nn.Sequential(S3D_G_block(480,[192,96,208,16,48,64]),
                                      S3D_G_block(512, [160, 112, 224, 24, 64, 64]),
                                      S3D_G_block(512, [128, 128, 256, 24, 64, 64]),
                                      S3D_G_block(512, [112, 144, 288, 32, 64, 64]),
                                      S3D_G_block(528, [256, 160, 320, 32, 128, 128])) #16，14，14
        self.pool4=nn.MaxPool3d(kernel_size=(2,2,2),stride=2) #8，7，7
        self.Inception3=nn.Sequential(S3D_G_block(832,[256,160,320,32,128,128]),
                                      S3D_G_block(832, [384, 192, 384, 48, 128, 128]))# 8，7，7 共1024channel
        self.avg_pool=nn.AvgPool3d(kernel_size=(8,7,7)) 
        self.dropout = nn.Dropout(0.4)
        self.linear=nn.Linear(1024,num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.Inception1(x)
        x = self.pool3(x)
        x = self.Inception2(x)
        x = self.pool4(x)
        x = self.Inception3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1) #(:, 1024)
        if self.return_feature:
            return x
        x = self.dropout(x)
        return self.linear(x) #(batch_size, num_class)