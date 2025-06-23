import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .module.GAL.gal import GAL
from .module.video_swin import SwinTransformerBlock3D
from .module.dtum import Res_CBAM_block





class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)  # [1, 3, 512, 512]
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]  #dark3-5: [1, 128, 64, 64] [1, 256, 32, 32] [1, 512, 16, 16]

        #-------------------------------------------#
        #   [1, 512, 16, 16] -> [1, 256, 16, 16]
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  [1, 256, 16, 16] -> [1, 256, 32, 32]
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  [1, 256, 32, 32] + [1, 256, 32, 32] -> [1, 512, 32, 32]
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #  [1, 512, 32, 32] -> [1, 256, 32, 32]
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #  [1, 256, 32, 32] -> [1, 128, 32, 32]
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #  [1, 128, 32, 32] -> [1, 256, 64, 64]
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #  [1, 128, 64, 64] + [1, 128, 64, 64] -> [1, 256, 64, 64]
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   [1, 256, 64, 64] -> [1, 128, 64, 64]
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        return P3_out

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs



class Neck(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        )
        self.conv_cur = BaseConv(channels[0], channels[0],3,1)
        

        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        
        self.gal = GAL(128)
        self.conv_fin_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1),
        )
        self.conv_fre_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1),
        )

        self.nolocal = NonLocalBlock(128)
        self.swin = SwinTransformerBlock3D(128,num_frames=self.num_frame+1)
        self.conv_t = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame+1), channels[2]*4,3,1),
            BaseConv(channels[2]*4,channels[0]*2,3,1),
            BaseConv(channels[2]*2,channels[0],3,1)
        )

        self.conv_fre = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

        self.conv_fre_new = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame+1), channels[0]*4,3,1),
            BaseConv(channels[0]*4,channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

        ####memory####
        self.keyvalue_Q = KeyValue_Q(128,64,128)
        self.keyvalue_M = KeyValue_M(128,64,128)
        self.memory = MemoryReader()
        self.fuse = fuse(128,128)
        self.resblock0 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock1 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock2 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])


        # Wavelet Transform
        self.conv_dw = WTConv2d(128,128,kernel_size=5,wt_levels=3)

        self.conv_t = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame+1), channels[2]*2,3,1),
            BaseConv(channels[2]*2,channels[0],3,1)
        )

        # temporal
        self.tar = TARFusion(num_feat= 128, num_frame= self.num_frame,center_frame_idx=self.num_frame-1)





    def forward(self, feats):
        all_feats = []   # 5*[2,128,64,64]
        s_cat_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)  
        s_cov_feat = self.conv_ref(s_cat_feat) 
        s_feat = self.conv_cur(s_cov_feat*feats[-1]) 
        s_feat = self.conv_cr_mix(torch.cat([s_feat, feats[-1]], dim=1)) 
        s_feat = self.nolocal(s_feat)


        K_Q, V_Q = self.keyvalue_Q(feats[-1])
        K_M, V_M = self.keyvalue_M(s_feat)
        s_feat = self.memory(K_M, V_M, K_Q, V_Q)
        
        # temporal
        s_stk_feat = torch.stack(feats,dim=1)
        s_stk_t = self.tar(s_stk_feat)  
        s_feat_l = self.gal(s_stk_t) 

        s_feat_g =self.conv_fin_mix(torch.cat([s_feat,s_feat_l],dim=1))
        
    
        # frequency
        fre_feats = []
        for i in range(self.num_frame):
            f_dw_feat = self.conv_dw(feats[i])  
            fre_feats.append(f_dw_feat)

        f_cat_feat = torch.cat([fre_feats[i] for i in range(self.num_frame)], dim=1)  
        

        ft_feat = torch.cat([f_cat_feat,s_feat_l],dim=1)
        pt_feat = self.conv_fre_new(ft_feat) 
        ft_feat_g = self.swin(ft_feat)  
        ft_feat_g = self.conv_t(ft_feat_g)

       
        feat_rcu1 = self.resblock0(torch.cat([s_feat_g,pt_feat],dim=1))
        feat_rcu2 = self.resblock1(torch.cat([pt_feat,ft_feat_g],dim=1))
        feat_rcu3 = self.resblock2(torch.cat([feat_rcu1,feat_rcu2],dim=1))

        all_feats.append(feat_rcu3)

        return all_feats


class slowfastnet(nn.Module):
    def __init__(self, num_classes, num_frames=5):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = YOLOPAFPN(0.33,0.50)   
        self.neck = Neck(channels=[128,256,512], num_frame=num_frames)
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")

    def forward(self, x):

        # [2, 3, 5, 512, 512]
        B, C, T, H, W = x.shape
        feat = []
        for t in range(T):
            # [2,3,512,512] -> [2,128,64,64]
            feat.append(self.backbone(x[:, :, t]))

        if self.neck:
            fused_features = self.neck(feat)
        outputs = self.head(fused_features)
        
        return outputs







class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio 
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x 
        return out



#  F3Net
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m
def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]
def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        # print("ce",x.shape)   # [2, 128, 64, 64]
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]  [2, 128, 64, 64]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out
    



#  Motion-memory
class Con1x1WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con1x1WithBnRelu, self).__init__()
        self.con1x1 = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con1x1(input)))
class KeyValue_Q(torch.nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_Q, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)
class KeyValue_M(torch.nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_M, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)
class MemoryReader(torch.nn.Module):
    def __init__(self):
        super(MemoryReader, self).__init__()
        self.memory_reduce = Con1x1WithBnRelu(256, 128)

    def forward(self, K_M, V_M, K_Q, V_Q): 
        B, C_K, H, W = K_M.size()
        _, C_V, _, _ = V_M.size()

        K_M = K_M.view(B, C_K,  H * W)
        K_M = torch.transpose(K_M, 1, 2) 
        K_Q = K_Q.view(B, C_K, H * W) 

        w = torch.bmm(K_M, K_Q) 
        w = w / math.sqrt(C_K)
        w = F.softmax(w, dim=1)
        V_M = V_M.view(B, C_V,  H * W) 

        mem = torch.bmm(V_M, w)
        mem = mem.view(B, C_V, H, W)

        E_t = torch.cat([mem, V_Q], dim=1) 

        return self.memory_reduce(E_t)
    

class fuse(torch.nn.Module):
    def __init__(self, indim_h, indim_l):
        super(fuse, self).__init__()
        self.conv_h = torch.nn.Conv2d(indim_h, 1, kernel_size=1)
        self.conv_l = torch.nn.Conv2d(indim_l, indim_h, kernel_size=3, padding=1, stride=1)
        self.fc = torch.nn.Linear(128, 128)

    def forward(self, l, h):
        S = self.conv_h(h)* self.conv_l(l) 
        temp = F.adaptive_avg_pool2d(S, (1, 1))
        temp = temp.squeeze(-1).squeeze(-1)
        c = self.fc(temp)
        c = c.unsqueeze(-1).unsqueeze(-1)
        return S + c * S
    
   

#  WTConv
import pywt
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x



#  FBANet
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Downsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample_flatten, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv(x).contiguous()  
        return out

    def flops(self, H, W):
        flops = 0
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

class Upsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample_flatten, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.deconv(x).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops
    


class TARFusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=4):
        super(TARFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        self.downsample1 = Downsample_flatten(num_feat, num_feat*2)
        self.downsample2 = Downsample_flatten(num_feat*2, num_feat*4)

        self.upsample1 = Upsample_flatten(num_feat*4, num_feat*2)
        self.upsample2 = Upsample_flatten(num_feat*4, num_feat)

        n_resblocks = 2
        conv = default_conv
        m_res_block1 = [
            ResBlock(
                conv, num_feat, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block2 = [
            ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block3 = [
            ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block4 = [
            ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block5 = [
            ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_fusion_tail = [conv(num_feat*2, num_feat, kernel_size=3)]

        self.res_block1 = nn.Sequential(*m_res_block1)
        self.res_block2 = nn.Sequential(*m_res_block2)
        self.res_block3 = nn.Sequential(*m_res_block3)
        self.res_block4 = nn.Sequential(*m_res_block4)
        self.res_block5 = nn.Sequential(*m_res_block5)
        self.fusion_tail = nn.Sequential(*m_fusion_tail)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()   # 2 5 128 64 64
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())  # [2, 128, 64, 64]
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))  # [10, 128, 64, 64]
        embedding = embedding.view(b, t, -1, h, w)  # [b,t,c,h,w]
        corr_diff = []
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1).unsqueeze(1)  # [b,1,h,w]
            corr_l.append(corr)

        for i in range(t):
            if i == self.center_frame_idx:    
                continue
            else:
                corr_difference = torch.abs(corr_l[i] - corr_l[self.center_frame_idx])  
                corr_diff.append(corr_difference)
        
        corr_l_cat = torch.cat(corr_l, dim=1)

        corr_prob = torch.sigmoid(torch.cat(corr_diff, dim=1))  # [b,t-1,h,w]
        
        corr_prob = corr_prob.unsqueeze(2).expand(b, t-1, c, h, w)  # [b,t,c,h,w]
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # [b,(t-1)*c,h,w]

        aligned_oth_feat = aligned_feat[:, :4, :, :, :]      
        aligned_oth_feat = aligned_oth_feat.view(b, -1, h, w) * corr_prob
        
        aligned_feat_guided = torch.zeros(b, t*c, h, w).to('cuda')
        aligned_feat_guided[:, 0 : c, :, :] = aligned_feat[:, 4 : 5, :, :, :].view(b, -1, h, w) 
        aligned_feat_guided[:, c:, :, :] = aligned_oth_feat
        feat = self.lrelu(self.feat_fusion(aligned_feat_guided)) 

        # Hourglass 
        feat_res1 = self.res_block1(feat)
        down_feat1 = self.downsample1(feat_res1)
        feat_res2 = self.res_block2(down_feat1)
        down_feat2 = self.downsample2(feat_res2)

        feat3 = self.res_block3(down_feat2)

        up_feat3 = self.upsample1(feat3)
        concat_2_1 = torch.cat([up_feat3, feat_res2], 1)
        feat_res4 = self.res_block4(concat_2_1)
        up_feat4 = self.upsample2(feat_res4)
        concat_1_0 = torch.cat([up_feat4, feat_res1], 1)
        feat_res5 = self.res_block5(concat_1_0)

        feat_out = self.fusion_tail(feat_res5) + feat

        return feat_out











if __name__ == "__main__":
    
    # from yolo_training import YOLOLoss
    net = slowfastnet(num_classes=1, num_frame=5)
    # yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])
    a = torch.randn(4, 5,128,64,64)
    out = net(a)  
    print(out.shape)

