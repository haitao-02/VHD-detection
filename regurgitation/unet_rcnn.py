import torch
import torch.nn as nn
import torch.nn.functional as F

def init_conv_weights(layer):
    nn.init.normal_(layer.weight.data, std=0.01)
    nn.init.constant_(layer.bias.data, val=0)
    return layer

def conv1x1(in_channel, out_channel, **kwargs):
    layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, **kwargs)
    init_conv_weights(layer)
    return layer

def conv3x3(in_channel, out_channel, **kwargs):
    layer = nn.Conv2d(in_channel, out_channel, kernel_size=3, **kwargs)
    init_conv_weights(layer)
    return layer

class DownSampleLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dilation=1):
        super(DownSampleLayer, self).__init__()
        self.out_chn = out_chn
        self.conv = nn.Sequential(
                conv3x3(in_chn, out_chn, dilation=1, padding=1),
                nn.BatchNorm2d(out_chn),
                nn.ReLU(),
                conv3x3(out_chn, out_chn, dilation=dilation, padding=dilation),
                nn.BatchNorm2d(out_chn),
                nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        return (self.pool(x), x)

class UpSampleLayer(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(UpSampleLayer, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Sequential(
                conv3x3(2*out_chn, out_chn, dilation=1, padding=1), 
                nn.BatchNorm2d(out_chn),
                nn.ReLU(),
                conv3x3(out_chn, out_chn, dilation=1, padding=1),
                nn.BatchNorm2d(out_chn),
                nn.ReLU())

    def forward(self, x, x_skip):
        x = self.tconv(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv(x)
        return (x)

class UnetRCNN(nn.Module):
    def __init__(self, num_classes=3, win_size=32, features=32, dilation=2, depth=4, **kwargs):
        super(UnetRCNN, self).__init__()
        self.num_classes = num_classes
        self.win_size = win_size
        self.down_sample, self.up_sample = [], []
        self.dconv = []
        self.dilation = dilation
        for i in range(depth):
            if i == 0:
                self.down_sample += [DownSampleLayer(3, features, dilation=dilation)]
            else:
                self.down_sample += [DownSampleLayer(features, features*2, dilation=dilation)]
                features *= 2
        self.down_sample = nn.Sequential(*self.down_sample)
        d = [1,2]
        for i in range(len(d)):
            self.dconv += [nn.Sequential(
                conv3x3(features, features, padding=d[i], dilation=d[i]), 
                nn.BatchNorm2d(features),
                nn.ReLU())]
        self.dconv = nn.Sequential(*self.dconv)
        for i in range(depth):
            if i == 0:
                self.up_sample += [UpSampleLayer(features, features)]
                features //= 2
            else:
                self.up_sample += [UpSampleLayer(2*features, features)]
                features //= 2
        self.up_sample = nn.Sequential(*self.up_sample)
        self.conv_s = nn.Sequential(
            conv1x1(features*2, num_classes),
            nn.Sigmoid()
            )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)
        self.detector = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
        self.classifier = nn.Linear(256, 2)
    def forward(self, x, seg_slice):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        seg_slice = seg_slice.view(-1)
        skips = []
        for i in range(len(self.down_sample)):
            x, x_skip = self.down_sample[i](x)
            skips.append(x_skip[seg_slice])
        for i in range(len(self.dconv)):
            x = self.dconv[i](x)

        xs = x[seg_slice]

        un = len(self.up_sample)
        for i in range(un):
            xs = self.up_sample[i](xs, skips[un-1-i])
        
        out_seg = self.conv_s(xs)
 
        x = self.avgpool(x)
        x = x.view(b, t, -1)

        # self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        # self.rnn.flatten_parameters()

        out_cls = self.classifier(torch.mean(x, dim=1))
        out_kf = self.detector(x).view(-1, )
        return (out_seg, out_cls, out_kf)

if __name__ == "__main__":
    x = torch.randn(2,3,3,128,128)
    model = UnetRCNN()
    y = model(x, torch.Tensor([1,0,1,0,1,0]).byte())
    print(y[0].size(), y[1].size(), y[2].size())