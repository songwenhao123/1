# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dualselfatt import CAM_Module

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CAM(nn.Module):
    def __init__(self, all_channel):
        super(CAM, self).__init__()
        #self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        # self-channel attention
        self.cam = CAM_Module(all_channel)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.cam(summation_sa)

        return sc_feat

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class DRDB(nn.Module):
    def __init__(self, in_ch, growth_rate, out_channels):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=1, dilation=1)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=3, dilation=3)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=5, dilation=5)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=7, dilation=7)
        in_ch_ += growth_rate
        self.sobelconv=Sobelxy(in_ch)
        print(in_ch_,in_ch)
        self.conv = nn.Conv2d(in_ch_, out_channels, 1, padding=0)
        self.convup =Conv1(in_ch,out_channels)
        

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = self.sobelconv(x)
        sobel=self.convup(out)
        
        return F.leaky_relu(x6+sobel,negative_slope=0.1)

class Ab_DRDB(nn.Module):
    def __init__(self, in_ch, out_channels):
        """
        消融DRDB的替代模块。
        
        Args:
            in_ch (int): 输入通道数。
            growth_rate (int): DenseNet中每层网络的增长率。
            out_channels (int): 输出通道数。
        
        Returns:
            None
        
        """
        super(Ab_DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
     
        self.Dcov2 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.Dcov3 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.Dcov4 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
       
        self.Dcov5 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.conv = nn.Conv2d(in_ch_, out_channels, 1, padding=0)
        self.convup =Conv1(in_ch,out_channels)
        

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)

        x6 = self.conv(x5)
        
        return F.leaky_relu(x6,negative_slope=0.1)

class Ab_SIM(nn.Module):
    def __init__(self, in_ch, out_channels):
        """
        消融SIM的替代模块。
        
        Args:
            in_ch (int): 输入通道数。
            growth_rate (int): DenseNet中每层网络的增长率。
            out_channels (int): 输出通道数。
        
        Returns:
            None
        
        """
        super(Ab_SIM, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
     
        self.Dcov2 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.Dcov3 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.Dcov4 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
       
        self.Dcov5 = nn.Conv2d(in_ch_, in_ch_, 3, padding=1)
      
        self.conv = nn.Conv2d(in_ch_, out_channels, 1, padding=0)
        self.convup =Conv1(in_ch,out_channels)
        

    def forward(self, x, y):
        out1 = torch.cat((x, y), dim=1)
        x1 = self.Dcov1(out1)
        x1 = F.relu(x1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)

        x6 = self.conv(x5)
        
        return F.leaky_relu(x6,negative_slope=0.1)

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

class Fusion_DRDB(nn.Module):
    def __init__(self, output):
        super(Fusion_DRDB, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

class Fusion_DRDB_CAM(nn.Module):
    def __init__(self, output):
        super(Fusion_DRDB_CAM, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x


class Baseline(nn.Module):
    def __init__(self, output):
        super(Baseline, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=ConvLeakyRelu2d(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = ConvLeakyRelu2d(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = ConvLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = ConvLeakyRelu2d(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out

class LFE(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(LFE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, 48, kernel_size=1)
        self.ca = ChannelAttention(48)

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = max_out
        x2 = self.conv1(x2)
        att2 = self.sigmoid(x2+x1)
        out = torch.mul(x1, att2) + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout
    
class DRDB_LFE(nn.Module):
    def __init__(self, output):
        super(DRDB_LFE, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = LFE(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class EM(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(EM, self).__init__()
        self.conv = nn.Conv2d(inchannel*2, inchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.ca = ChannelAttention(inchannel)
        self.sa = SpAttention()
        self.rconv0 = nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
        self.rbn = nn.BatchNorm2d(inchannel)

        self.convfinal = nn.Conv2d(inchannel, outchannel, kernel_size=1)

    def forward(self, laster, current):

        out1 = torch.cat((laster, current), dim=1)
        out1 = self.conv(out1)

        x1 = laster * out1
        ir1 = current * out1
        f = x1 + ir1
        f = self.rconv(f)
        ca = self.ca(f)
        ca_f = f.mul(ca)
        
        sa = self.sa(ca_f)
        sa_f = ca_f.mul(sa)

        f = self.rbn(self.rconv0(sa_f))
        
        f = f + laster

        f = self.convfinal(f)

        return f

class DRDB_CAM_SIM_s(nn.Module):
    def __init__(self, output):
        super(DRDB_CAM_SIM_s, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 64)
        self.sim = EM(64, 64)
        self.output1 = ConvBnLeakyRelu2d(64, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x

class DRDB_CAM_SIM(nn.Module):
    def __init__(self, output):
        super(DRDB_CAM_SIM, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = EM(9, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x
###################################################################(Ablation)#########################################################################################

class WO_Seg(nn.Module):
    def __init__(self, output):
        """
        审稿人1 wo_seg类
        
        Args:
            output (int): 输出通道数
        
        Returns:
            None
        
        """
        super(WO_Seg, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        # x=self.sim(x)
        x=self.output1(x)
        x=self.output(x)
        return x


class WO_SIM(nn.Module):
    def __init__(self, output):
        """
        审稿人wo_SIM类
        
        Args:
            output (int): 输出通道数
        
        Returns:
            None
        
        """
        super(WO_SIM, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = Ab_SIM(18, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x

class WO_DRDB(nn.Module):
    def __init__(self, output):
        
        super(WO_DRDB, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=ConvLeakyRelu2d(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = ConvLeakyRelu2d(vis_ch[1],  vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = ConvLeakyRelu2d(inf_ch[0],  inf_ch[1])
        self.inf_rgbd2 = ConvLeakyRelu2d(inf_ch[1],  inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = EM(9, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x
    
class R1_DRDB(nn.Module):
    def __init__(self, output):
        """
        审稿人要求WO_DRDB
        
        Args:
            output (int): 输出通道数
        
        Returns:
            None
        """
        super(R1_DRDB, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=Ab_DRDB(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = Ab_DRDB(vis_ch[1],  vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = Ab_DRDB(inf_ch[0],  inf_ch[1])
        self.inf_rgbd2 = Ab_DRDB(inf_ch[1],  inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        self.fuse = CAM(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = EM(9, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(self.fuse(x_vis_p2,x_inf_p2))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x


device='cuda'   
def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2
   
class WO_CAM(nn.Module):
    def __init__(self, output, addition_mode='Sum'):
        super(WO_CAM, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        growth_rate = [8, 16]
        output=1
        self.addition_mode = addition_mode
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=DRDB(vis_ch[0], growth_rate[0], vis_ch[1])
        self.vis_rgbd2 = DRDB(vis_ch[1], growth_rate[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = DRDB(inf_ch[0], growth_rate[0], inf_ch[1])
        self.inf_rgbd2 = DRDB(inf_ch[1], growth_rate[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.fuse = ConvLeakyRelu2d(inf_ch[2])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = EM(9, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir, mask):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        if self.addition_mode=='Sum':      
            x = x_vis_p2+x_inf_p2
        elif self.addition_mode=='Average':
            x=(x_vis_p2+x_inf_p2)/2         
        elif self.addition_mode=='l1_norm':
            x=l1_addition(x_vis_p2,x_inf_p2)
        else:
            print('Wrong!')
        
        x=self.decode4(x)
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        x=self.sim(x,mask)
        x=self.output1(x)
        x=self.output(x)
        return x
def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
