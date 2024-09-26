import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个layer_contrast模块，用于对特征进行对比度增强
class layer_contrast(nn.Module):
    def __init__(self):
        super(layer_contrast, self).__init__()

    def forward(self, feature_shuffle):
        # feature_shuffle是一个PyTorch张量，表示融合后的特征
        # 计算feature_shuffle的平均值，作为mean_vector
        mean_vector = torch.mean(feature_shuffle, [1, 2], keepdim=True)
        # 计算feature_shuffle与mean_vector的平方差的平均值，作为feature_contrast
        feature_contrast = torch.mean((feature_shuffle - mean_vector) ** 2, [1, 2], keepdim=True).sqrt()
        # 计算feature_contrast的平均值，作为contrast_vector
        contrast_vector = torch.mean(feature_contrast, [1, 2], keepdim=True)
        # 使用contrast_vector对feature_shuffle进行放缩，得到feature_fusion_enhancement
        feature_fusion_enhancement = contrast_vector * feature_shuffle
        return feature_fusion_enhancement



# 定义 CMDAF 模块
def CMDAF(vi_feature, ir_feature):
    # 定义 sigmoid 函数
    sigmoid = nn.Sigmoid()
    # 定义全局平均池化层
    gap = nn.AdaptiveAvgPool2d(1)
    # 获取可见光和红外光特征的尺寸
    batch_size, channels, _, _ = vi_feature.size()

    # 跨模态自注意力机制
    # 计算可见光特征之间的相似度矩阵
    vi_att = torch.matmul(vi_feature.view(batch_size, channels, -1), vi_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    # 计算红外光特征之间的相似度矩阵
    ir_att = torch.matmul(ir_feature.view(batch_size, channels, -1), ir_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    # 计算可见光和红外光特征之间的相似度矩阵
    cross_att = torch.matmul(vi_feature.view(batch_size, channels, -1), ir_feature.view(batch_size, channels, -1).permute(0, 2, 1))
    # 对相似度矩阵进行 softmax 归一化
    vi_att = F.softmax(vi_att, dim=-1)
    ir_att = F.softmax(ir_att, dim=-1)
    cross_att = F.softmax(cross_att, dim=-1)
    # 通过相似度矩阵对可见光和红外光特征进行加权和融合
    vi_feature = torch.matmul(vi_att, vi_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _) + torch.matmul(cross_att, ir_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _)
    ir_feature = torch.matmul(ir_att, ir_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _) + torch.matmul(cross_att.permute(0, 2, 1), vi_feature.view(batch_size, channels, -1)).view(batch_size, channels, _, _)

    # 模态选择机制
    # 通过全局平均池化层计算可见光和红外光特征的通道权重
    vi_weight = gap(vi_feature)
    ir_weight = gap(ir_feature)
    # 对通道权重进行 softmax 归一化
    vi_weight = F.softmax(vi_weight, dim=1)
    ir_weight = F.softmax(ir_weight, dim=1)
    # 通过通道权重对可见光和红外光特征进行加权
    vi_feature = vi_feature * vi_weight
    ir_feature = ir_feature * ir_weight

    # 多尺度特征融合机制
    # 通过双线性插值将可见光和红外光特征上采样到原始尺寸
    # vi_feature = F.interpolate(vi_feature, scale_factor=2, mode='bilinear', align_corners=True)
    # ir_feature = F.interpolate(ir_feature, scale_factor=2, mode='bilinear', align_corners=True)
    # 将可见光和红外光特征在通道维度上拼接
    feature = torch.cat([vi_feature, ir_feature], 1)

    # 返回融合后的特征
    return feature

