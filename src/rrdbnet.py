import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ChannelAttention(nn.Module):
    """Channel attention module with reduction."""
    def __init__(self, num_feat: int, reduction: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, bias=True),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)

class LightRRDB(nn.Module):
    """Fixed Lightweight Residual-in-Residual Dense Block"""
    def __init__(self, in_channels: int, growth_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(growth_channels, growth_channels, 3, 1, 1, groups=growth_channels),
            nn.Conv2d(growth_channels, growth_channels, 1)
        )
        self.attention = ChannelAttention(growth_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(growth_channels, in_channels, 1, 1, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.attention(out)
        out = self.conv_out(out)
        return identity + out * 0.2

class RRDB(nn.Module):
    """Residual-in-Residual Dense Block with configurable architecture."""
    def __init__(self, num_feat: int, num_grow_ch: int = 32, use_light_block: bool = False):
        super().__init__()
        self.rdb1 = LightRRDB(num_feat, num_grow_ch) if use_light_block else self._make_dense_block(num_feat, num_grow_ch)
        self.rdb2 = LightRRDB(num_feat, num_grow_ch) if use_light_block else self._make_dense_block(num_feat, num_grow_ch)
        self.rdb3 = LightRRDB(num_feat, num_grow_ch) if use_light_block else self._make_dense_block(num_feat, num_grow_ch)
        
    def _make_dense_block(self, num_feat: int, num_grow_ch: int) -> nn.Sequential:
        layers = []
        in_channels = num_feat
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels, num_grow_ch, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
            in_channels += num_grow_ch
        
        layers.append(nn.Conv2d(in_channels, num_feat, 1, 1, 0))
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2

class RRDBNet(nn.Module):
    """Enhanced RRDBNet architecture with configurable options."""
    
    def __init__(self, 
                 num_in_ch: int = 3, 
                 num_out_ch: int = 3, 
                 scale: int = 4, 
                 num_feat: int = 64, 
                 num_block: int = 23, 
                 num_grow_ch: int = 32, 
                 use_light_blocks: bool = False,
                 progressive_scale: bool = False):
        super().__init__()
        self.scale = scale
        self.progressive_scale = progressive_scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.body = nn.ModuleList([
            RRDB(num_feat, num_grow_ch, use_light_blocks) 
            for _ in range(num_block)
        ])
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        if progressive_scale:
            self.upsample = nn.Sequential()
            current_scale = 1
            while current_scale < scale:
                self.upsample.append(nn.Sequential(
                    nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ))
                current_scale *= 2
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1) if scale == 4 else nn.Identity(),
                nn.PixelShuffle(2) if scale == 4 else nn.Identity(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True) if scale == 4 else nn.Identity()
            )
        
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv_first, self.conv_body, self.conv_last]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.lrelu(self.conv_first(x))
        
        for block in self.body:
            feat = block(feat)
        
        feat = self.conv_body(feat) + feat
        
        if self.progressive_scale:
            out = feat
            for upsample_layer in self.upsample:
                out = upsample_layer(out)
        else:
            out = self.upsample(feat)
        
        out = self.conv_last(out)
        return out