import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from .kan_layers.kan_linear import KANLinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class SpectralResidualBlock(nn.Module):
    def __init__(self, dim):
        super(SpectralResidualBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,1,1)
        )
        self.ca = CALayer(dim)
    def forward(self, x):
        res = x
        x = self.convs(x) 
        x = self.ca(x)
        return x + res
    
class SpatialResBlock(nn.Module):
    def __init__(self,dim):
        super(SpatialResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,3,1,1)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x
    

class DWConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DWConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels,in_channels,3,1,1,groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels,out_channels,1,1,0)
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.point_conv(x)
        return x

class SpaKAN(nn.Module):
    def __init__(self,dim,image_size=64):
        super(SpaKAN, self).__init__()
        self.dim = dim
        self.conv3x3 = nn.Conv2d(dim,dim,3,1,1)
        self.conv5x5 = nn.Conv2d(dim,dim,5,1,2)
        self.dwconv = DWConv(dim * 2,1)
        self.kan1 =  nn.Linear(1,dim)
        self.kan2 =  nn.Linear(dim,1)
        
    def forward(self, x):
        b,c,h,w = x.shape
        feat1 = F.relu(self.conv5x5(x))
        feat2 = F.relu(self.conv3x3(feat1))
        feat = torch.cat([feat1,feat2],dim=1)
        feat = F.relu(self.dwconv(feat))
        feat = rearrange(feat,'b c h w -> b (h w) c')
        feat = self.kan1(feat)
        feat = self.kan2(feat)
        feat = rearrange(feat,'b (h w) c -> b c h w',h=h,w=w)
        return x * feat + x
    
    
class SpeKAN(nn.Module):
    def __init__(self,dim,scale):
        super(SpeKAN, self).__init__()
        self.dim = dim
        self.scale = scale
        self.avg_kan01 =  KANLinear(dim)
        self.avg_kan02 = KANLinear(dim)
        self.max_kan01 = KANLinear(dim)
        self.max_kan02 =  KANLinear(dim)
        self.sum_kan = KANLinear(dim)
        
        
    def forward(self, x):
        avg_score = F.adaptive_avg_pool2d(x,(1,1))
        avg_score = rearrange(avg_score,'b c 1 1 -> b c')
        avg_score = self.avg_kan01(avg_score)
        avg_score = self.avg_kan02(avg_score)
        max_score = F.adaptive_max_pool2d(x,(1,1))
        max_score = rearrange(max_score,'b c 1 1 -> b c')
        max_score = self.max_kan01(max_score)
        max_score = self.max_kan02(max_score)
        score = self.sum_kan(torch.cat([avg_score,max_score],dim=1))
        score = rearrange(score,'b c -> b c 1 1')
        x =  x * score + x
        return x

        

class KSSANet(nn.Module):
    def __init__(self,hsi_bands,scale,depth,dim):
        super(KSSANet, self).__init__()
        self.dim = dim
        self.depth = depth
        self.hsi_bands = hsi_bands
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(hsi_bands,dim,kernel_size=3,padding=1,stride=1),
        )
        self.spatial_blocks = nn.ModuleList([SpaKAN(dim,image_size=64) for _ in range(depth)])
        self.spectral_blocks = nn.ModuleList([SpeKAN(dim,scale) for _ in range(depth)])
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, hsi_bands,3,1,1)
        )
        
    def forward(self, x):
        up_hsi  = F.interpolate(x,scale_factor=self.scale,mode='bicubic',align_corners=True)
        b,c,h,w = up_hsi.shape
        feat  = self.head(up_hsi)
        for i in range(self.depth):
            feat = self.spatial_blocks[i](feat)
            feat = self.spectral_blocks[i](feat)
        feat = self.tail(feat)
        return feat + up_hsi