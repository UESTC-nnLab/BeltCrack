import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        
        if self.activation is not None:
            out = self.act(out)
            
        return out


class TDM_S(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5, nres_b=1):
        super(TDM_S, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta

        base_filter = 128  # bf

        self.feat0 = ConvBlock(128, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 对目标帧特征提取：h*w*3-->h*w*base_filter
        self.feat_diff = ConvBlock(128, 64, 3, 1, 1, activation='prelu', norm=None)  # 对rgb的残差信息进行特征提取：h*w*3 --> h*w*64

        self.conv1 = ConvBlock((self.nframes-1)*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 对pooling后堆叠的diff特征增强

        # Res-Block1,h*w*bf-->h*w*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body1.append(ConvBlock(base_filter, 128, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block1,h*w*bf-->H*W*64，对第一次补充的目标帧特征增强
        modules_body2 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body2.append(ConvBlock(base_filter, 128, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍
    
    def forward(self, neigbor):
        frame_list = neigbor
        rgb_diff = []
        for i in range(self.nframes-1):
            rgb_diff.append(frame_list[i] - frame_list[i+1])
        rgb_diff = torch.stack(rgb_diff, dim=1)
        B, N, C, H, W = rgb_diff.size()  # [1, nframe-1, 3, 160, 160]

        # 对目标帧及残差图进行特征提取
        lr_f0 = self.feat0(frame_list[-1])  # h*w*3 --> h*w*256   4 128 64 64
         
        diff_f = self.feat_diff(rgb_diff.view(-1, C, H, W)) #16 64 64 64 

        down_diff_f = self.avg_diff(diff_f).view(B, N, -1, H//2, W//2)  # 每个diff特征，被降采样2倍[4，4,64,32,32]
        stack_diff = []
        for j in range(N):
            stack_diff.append(down_diff_f[:, j, :, :, :])
        stack_diff = torch.cat(stack_diff, dim=1)
        stack_diff = self.conv1(stack_diff)  # diff 增强 4 128 32 32 

        up_diff1 = self.res_feat1(stack_diff)  # 先过卷积256--》64再上采样  4 64 32 32

        up_diff1 = F.interpolate(up_diff1, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64   4 64 64 64
        up_diff2 = F.interpolate(stack_diff, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道还是256  4 128 64 64 

        compen_lr = self.apha * lr_f0 + self.belta * up_diff2   # 4 128 64 64 

        compen_lr = self.res_feat2(compen_lr)  # 第一次补偿后增强  4 64 64 64

        compen_lr = self.apha * compen_lr + self.belta * up_diff1

        return compen_lr
    # def forward(self, lr, neigbor):
    #     lr_id = self.nframes // 2
    #     neigbor.insert(lr_id, lr)  # 将中间目标帧插回去
    #     frame_list = neigbor
    #     rgb_diff = []
    #     for i in range(self.nframes-1):
    #         rgb_diff.append(frame_list[i] - frame_list[i+1])
    #     rgb_diff = torch.stack(rgb_diff, dim=1)
    #     B, N, C, H, W = rgb_diff.size()  # [1, nframe-1, 3, 160, 160]

    #     # 对目标帧及残差图进行特征提取
    #     lr_f0 = self.feat0(lr)  # h*w*3 --> h*w*256
    #     diff_f = self.feat_diff(rgb_diff.view(-1, C, H, W))

    #     down_diff_f = self.avg_diff(diff_f).view(B, N, -1, H//2, W//2)  # 每个diff特征，被降采样2倍[1，4,64,80,80]
    #     stack_diff = []
    #     for j in range(N):
    #         stack_diff.append(down_diff_f[:, j, :, :, :])
    #     stack_diff = torch.cat(stack_diff, dim=1)
    #     stack_diff = self.conv1(stack_diff)  # diff 增强

    #     up_diff1 = self.res_feat1(stack_diff)  # 先过卷积256--》64再上采样

    #     up_diff1 = F.interpolate(up_diff1, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64
    #     up_diff2 = F.interpolate(stack_diff, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道还是256

    #     compen_lr = self.apha * lr_f0 + self.belta * up_diff2

    #     compen_lr = self.res_feat2(compen_lr)  # 第一次补偿后增强

    #     compen_lr = self.apha * compen_lr + self.belta * up_diff1

    #     return compen_lr


import os
import torch.nn as nn
import torch.optim as optim
# from modules.RCAN_basic import *

import torch
# import model.arch_util as arch_util
import functools

from torchvision.transforms import *
import torch.nn.functional as F


# TDM-long
class TDM_L(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5):
        super(TDM_L, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 64

        self.compress_3 = ConvBlock(self.nframes*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 多尺度压缩3*3：h*w*3-->h*w*256
        # self.compress_5 = ConvBlock(self.nframes*64, base_filter, 5, 1, 2, activation='prelu', norm=None)  # 多尺度压缩5*5：h*w*3-->h*w*256
        # self.compress_7 = ConvBlock(self.nframes*64, base_filter, 7, 1, 3, activation='prelu', norm=None)  # 多尺度压缩7*7：h*w*3-->h*w*256

        self.conv1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减之前的嵌入
        self.conv2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减特征的增强
        self.conv3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减池化特征的增强

        self.conv4 = ConvBlock(base_filter, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)  # 相加之后的增强

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍
        self.sigmoid = nn.Sigmoid()

        self.fus = nn.Conv2d(base_filter*2, base_filter, 3, 1, 1)


    def forward(self, frame_fea_list):
        frame_fea = torch.cat(frame_fea_list, 1)  # [b nframe*64 h w]

        frame_list_reverse = frame_fea_list
        frame_list_reverse.reverse()  # [[B,64,h,w], ..., ]
        # multi-scale: 3*3
        # forward
        forward_fea3 = self.conv1(self.compress_3(torch.cat(frame_fea_list, 1)))
        # backward
        backward_fea3 = self.conv1(self.compress_3(torch.cat(frame_list_reverse, 1)))
        # 残差
        forward_diff_fea3 = forward_fea3 - backward_fea3

        backward_diff_fea3 = backward_fea3 - forward_fea3


        id_f3 = forward_diff_fea3  # [b 96 h w]
        id_b3 = backward_diff_fea3
        pool_f3 = self.conv3(self.avg_diff(forward_fea3))  # [b 96 h/2, w/2]
        up_f3 = F.interpolate(pool_f3, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64

        pool_b3 = self.conv3(self.avg_diff(backward_fea3))
        up_b3 = F.interpolate(pool_b3, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64

        enhance_f3 = self.conv2(forward_fea3)
        enhance_b3 = self.conv2(backward_fea3)

        f3 = self.sigmoid(self.conv4(id_f3 + enhance_f3 + up_f3))
        b3 = self.sigmoid(self.conv4(id_b3 + enhance_b3 + up_b3))
        att3 = f3 + b3
        module_fea3 = att3 * frame_fea + frame_fea


        return module_fea3

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class myNet(nn.Module):
    def __init__(self, nframes):
        super(myNet, self).__init__()
        # self.swin_out = swin_out  # 投影输出通道数
        self.nframes = nframes
        self.lr_idx = self.nframes // 2
        self.apha = 0.5
        self.belta = 0.5

        self.fea0 = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # 对视频帧特征提取：h*w*3-->h*w*64
        self.fea_all = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # 对视频帧特征提取：h*w*3-->h*w*64
        feature_extraction = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        self.res_feat_ext = nn.Sequential(*feature_extraction)

        self.tdm_s = TDM_S(nframes=self.nframes, apha=self.apha, belta=self.belta)
        self.tdm_l = TDM_L(nframes=self.nframes)
        self.fus = nn.Conv2d(64*self.nframes, 64, 3, 1, 1)
        self.msd = MSD()
        self.TSA_Fusion = TSA_Fusion(64, nframes=self.nframes, center=self.lr_idx)
        # self.swinir = SwinIR(swin_out=self.swin_out)
        # self.hat = HAT()
        # self.dconv = DeconvBlock(self.nframes*64, self.swin_out, 8, 4, 2, activation='prelu', norm=None)

        # Res-Block2,残差信息增强
        modules_body2 = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        # modules_body2.append(ConvBlock(self.swin_out, self.swin_out, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        #
        # # Res-Block3，downsample
        # modules_body3 = [
        #     ResnetBlock(self.swin_out, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
        #     for _ in range(1)]
        # modules_body3.append(ConvBlock(self.swin_out, 64, 8, 4, 2, activation='prelu', norm=None))
        # self.res_feat3 = nn.Sequential(*modules_body3)

        # reconstuction
        # self.conv_reconstuction = nn.Conv2d(self.swin_out*3, 3, 3, 1, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, neigbors):
        B, C, H, W = x.size()
        fea_x = self.fea0(x)
        # 首先所有帧输入短期TDM完成补偿
        compen_x = self.tdm_s(x, neigbors)
        # res0 = fea_x - compen_x
        # res0 = self.res_feat2(res0)
        # s_compen_x = fea_x + res0
        # s_compen_x = compen_x + fea_x
        s_compen_x = fea_x + compen_x

        frame_all = neigbors  # TDM_S中已经在neigbor中间插入了目标帧，此时neigbor就是全部帧
        feat_all = torch.stack(frame_all, dim=1)
        feat_all = self.fea_all(feat_all.view(-1, C, H, W))  # 【N 64 ps ps】
        feat_all = self.res_feat_ext(feat_all)
        feat_all = feat_all.view(B, self.nframes, -1, H, W)  # [B, N, 64, ps, ps]

        # 随后MSD配准
        aligned_fea = []
        ref_fea = feat_all[:, self.lr_idx, :, :, :]
        for i in range(self.nframes):
            neigbor_fea = feat_all[:, i, :, :, :]
            aligned_fea.append(self.msd(neigbor_fea, ref_fea))

        # TSA融合
        aligned_fea_all = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        fea = self.TSA_Fusion(aligned_fea_all)  # [b 64 ps ps]

        # 输入长期TDM完成长期多尺度特征补偿
        fram_fea_list = []  # 构建帧特征列表 [b 64 ps ps] * n
        for i in range(self.nframes):
            fram_fea_list.append(aligned_fea[i])  # 这里用配准后的特征可能会好点

        l_compen_x = self.tdm_l(fram_fea_list)  # [b 64*n ps ps]
        l_compen_x = self.fus(l_compen_x)  # [b 64 ps ps]

        # 残差增强
        res = fea - l_compen_x
        res = self.res_feat2(res)
        fea = fea + res

        # res2 = s_compen_x - fea
        # res2 = self.res_feat2(res2)
        # final_compen_x = s_compen_x + res2
        fea = fea + s_compen_x

        return fea
try:
    from DCNv2.dcn_v2 import DCN_sep  # 加载可变卷积，使用其他特征生成偏移量和蒙版
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class MSD(nn.Module):  # Ours
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=64, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea



if __name__ == "__main__":
    net = TDM_S(nframes=5)
    a = torch.rand(4,128,64,64)
    list = []
    for i in range(5):
        list.append(a)
    out = net(list)
    print(out.shape)