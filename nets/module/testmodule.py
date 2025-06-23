import torch
import torch.nn as nn
from two_stream_net.nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
class MSA(nn.Module):
    def __init__(self, channels=[128,256,512], num_frame=5, dim=4096):
        super().__init__()
        self.num_frame = num_frame
        self.K = nn.Sequential(
            # BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.V = nn.Sequential(
            # BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.Q = nn.Sequential(
            # BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.attn = nn.MultiheadAttention(embed_dim=4096, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, ref, cur):
        B, C, H, W = cur.shape
        K, V = self.K(ref), self.V(ref)
        Q = self.Q(cur)
        attn, _ = self.attn(Q.reshape(B,C,-1), K.reshape(B,C,-1), V.reshape(B,C,-1))
        attn = self.norm(attn+Q.reshape(B,C,-1))
        attn = self.norm(attn + self.ffn(attn)).reshape(B,C,H,W)
        return attn
    
if __name__ == "__main__":
    
    net = MSA([128],5,4096)
    a = torch.randn(4, 128,64,64)
    b = torch.randn(4, 128,64,64)
    out = net(a,b) 
    print(out.shape)