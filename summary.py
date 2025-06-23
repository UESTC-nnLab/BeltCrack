#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

# from two_stream_net.nets.testfile.slowfastnet import slowfastnet
from nets.yolo import YoloBody  # YOLOX对比方法
# from nets.cracknet6 import  slowfastnet

# crack 参数量实验
# from nets.cracknet5 import slowfastnet

if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 1
    phi         = 's'
    
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # m       = slowfastnet(num_classes, num_frames=5).to(device)
    m = YoloBody(num_classes,phi)
    # summary(m, (3, 5,input_shape[0], input_shape[1]))  # 注释掉就不会报错
    
    dummy_input     = torch.randn(1, 3, 5,input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
