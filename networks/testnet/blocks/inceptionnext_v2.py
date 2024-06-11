import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN


class InceptionNeXtBlock_V2(nn.Module):
    def __init__(self, dim, kernel_size=7, exp_rate=4, drop_path=0.):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        gc = int(dim * 0.2)

        ##vv2 5,11
        self.dwconv_hwd = nn.Conv3d(gc, gc, kernel_size=5, padding=2, groups=gc)
        self.dwconv_h = nn.Conv3d(gc, gc, kernel_size=(11, 11, 1), padding=(5, 5, 0), groups=gc)
        self.dwconv_w = nn.Conv3d(gc, gc, kernel_size=(1, 11, 11), padding=(0, 5, 5), groups=gc)
        self.dwconv_d = nn.Conv3d(gc, gc, kernel_size=(11, 1, 11), padding=(5, 0, 5), groups=gc)
        self.split_indexes = (dim - 4 * gc, gc, gc, gc, gc)
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, exp_rate * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(exp_rate * dim)
        self.pwconv2 = nn.Linear(exp_rate * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x_id, x_hwd, x_w, x_h, x_d = torch.split(x, self.split_indexes, dim=1) 
        x = torch.cat((x_id, self.dwconv_hwd(x_hwd), self.dwconv_w(x_w), self.dwconv_h(x_h), self.dwconv_d(x_d)), dim=1) 
        
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x
    
    
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )