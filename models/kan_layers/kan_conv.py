import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .kan_linear import KANLinear
from einops import rearrange

class KANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,bias=True,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kan_ops = nn.ModuleList([KANLinear((kernel_size**2),(kernel_size**2))
                                      for _ in range(out_channels * in_channels)])
        self.bias = bias
        if bias:
            self.bias_weight = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
            
        out = torch.zeros((batch_size, self.out_channels, out_height, out_width)).to(x.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for h in range(0, out_height):
                    for w in range(0, out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        x_slice = x[:, j, h_start:h_end, w_start:w_end]
                        x_slice = rearrange(x_slice, 'b h w -> b (h w)')
                        x_slice = self.kan_ops[i*self.in_channels+j](x_slice)
                        x_slice = rearrange(x_slice, 'b (h w) -> b h w', h=self.kernel_size, w=self.kernel_size)
                        out[:, i, h, w] += torch.sum(x_slice, dim=(1, 2))
            if self.bias:
                out[:, i, :, :] += self.bias_weight[i]
        return out
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64).cuda()
    model = KANConv2d(3, 64, 3, padding=1).cuda()
    y = model(x)
    print(y.shape)  