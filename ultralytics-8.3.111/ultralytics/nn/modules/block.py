# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from .cluster_utils import GetCluster, batched_index_select
from .activation import StarReLU
from .conv import Convolution
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "ClusterConv",
    "DyClusterConv2d",
    "DClusConv",
    "C3DClus"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the YOLOv8 mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (Tuple[int, int, int]): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    #相比传统的 SPP 模块，SPPF 通过串联多个相同尺寸的池化层来实现相似效果，但计算效率更高
    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # 将输入通道数c1减半到c_ = c1 // 2
        # 第一个1x1卷积，用于减少通道数
        self.cv1 = Conv(c1, c_, 1, 1)# 输入c1通道，输出c_通道，1x1卷积，步长1
        # 第二个1x1卷积，用于将拼接后的特征转换回目标通道数
        self.cv2 = Conv(c_ * 4, c2, 1, 1)# 输入c_*4通道(因为要拼接4个特征图)，输出c2通道
        # 最大池化层，使用给定的核大小k
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)# 保持特征图尺寸不变

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        # 第一步：通过1x1卷积减少通道数
        y = [self.cv1(x)]# 初始结果存入列表
        # 第二步：连续进行3次最大池化，每次池化结果都添加到列表中
        y.extend(self.m(y[-1]) for _ in range(3))# 每次池化都是对上一次池化结果进行操作
        # 第四步：通过1x1卷积调整通道数到目标输出c2
        return self.cv2(torch.cat(y, 1))# dim=1表示在通道维度拼接


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y

class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    #通过跨阶段连接和多分支特征融合，增强特征表达能力
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        #2 * self.c 确实是输出通道数，但它的核心作用是为后续的分支操作提供足够的通道维度，以便将特征分成两部分
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # nn.ModuleList 是一个存储 子模块 的容器类，用于管理动态生成或需要迭代访问的模块列表 nn.ModuleList 用于堆叠多个残差块
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1)) # 将特征在通道维度均分为两部分
        y.extend(m(y[-1]) for m in self.m) # 每个 Bottleneck 处理前一个输出
        return self.cv2(torch.cat(y, 1))# 拼接所有分支，调整通道数

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)# 指定分割的通道数
        y = [y[0], y[1]] # 转为列表便于操作
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        #n：Bottleneck模块的堆叠数量（默认1）
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # DCCONV
        self.cv1 = Conv(c1, c_, 1, 1) # 输入分支1的1x1卷积
        self.cv2 = Conv(c1, c_, 1, 1)## 输入分支2的1x1卷积
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)# 合并后的1x1卷积
        #2 * c_：输入通道数
        #c2：输出通道数
        #1：使用 1×1 卷积高效调整通道数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
        #nn.Sequential 是一个容器类，用于按顺序包装多个神经网络模块。它的主要作用是简化代码结构，让模型的前向传播过程更加清晰和简洁
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    #torch.cat((..., ...), 1)通道拼接
    #在通道维度（dim=1）上拼接两个分支的输出


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        #计算瓶颈层的中间通道数
        #c2：输出通道数
        #e：扩展系数（通常为 0.5，表示将通道数压缩一半）
        self.cv1 = Conv(c1, c_, k[0], 1)#1：步长（stride=1，保持特征图尺寸不变）
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)#定义第二个卷积层，将中间通道c_扩展回输出通道c2
        #g：分组卷积的组数（g = 1为普通卷积，g = c_为深度可分离卷积）（如 MobileNet 中的深度可分离卷积）
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    #


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x, w):
        """
        Passes input out unchanged.

        TODO: Update or remove?
        """
        return x

    def forward(self, x, w):
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # nn.ModuleList 是一个存储 子模块 的容器类，用于管理动态生成或需要迭代访问的模块列表 nn.ModuleList 用于堆叠多个残差块
    def forward(self, x):
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    #C3表示3个卷积层，K代表其中bottleneck中的卷积核为支持自定义
    #通过更多的分支夸层链接，丰富了模型的梯度流
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        #c3k这个参数为FALSE的时候，C3K2模块就是C2F模块
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
            #残差链接，允许网络跳过某些层，直接将输入添加到后续层的输出中
            网络可以学习 “残差”（即输入与目标输出之间的差异），而不是直接学习完整映射
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c1 = c1
        self.c2 = c2
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
            out = super().forward(x)
            return out


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    #C3k 继承自 C3，复用了其基本结构和前向传播逻辑
    #通过参数 k 允许用户自定义 Bottleneck 中的卷积核大小（默认 k=3）
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        #dim输入通道数
        #num_heads：注意力头的数量
        #attn_ratio：注意力中 Key 的维度占比，用于控制 Query/Key 的压缩程度
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        #成员变量初始化
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads#head_dim：每个注意力头处理的维度
        self.key_dim = int(self.head_dim * attn_ratio)#key_dim：每个注意力头的 Key/Query 的维度（通常小于或等于 head_dim）
        self.scale = self.key_dim**-0.5#scale：缩放因子，防止点积值太大导致梯度爆炸（标准做法）
        #网络模块
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        #self.qkv：一次性生成 Query、Key 和 Value
        #Query（Q）查询：当前要处理的位置/元素想知道什么 num_heads * key_dim
        #Key（K）：输入数据的特征标识 num_heads * key_dim
        #Value（V）：实际包含的信息内容 dim
        self.proj = Conv(dim, dim, 1, act=False)
        #self.proj：最后对注意力结果进行线性变换（1×1卷积），回到原始维度
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        #self.pe：位置编码，使用深度可分离卷积(g=dim)实现，不改变通道数
        #深度可分离卷积？

    # 向前传播
    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        #B Batch Size 批量大小，表示一次处理的样本数量（如图像数量）
        #C Channels 通道数，表示每个空间位置的特征维度（如RGB通道或特征图的深度）。
        #对于 RGB图像，C=3（红、绿、蓝三个通道）
        #对于 特征图（如卷积层的输出），C 表示该层提取的多种特征（如边缘、纹理、颜色等）
        #通常随着网络加深，通道数逐渐增加
        #H Height 特征图的高度（像素/网格行数）
        #W Width 特征图的宽度（像素/网格列数）

        N = H * W
        # N Num Positions 空间位置总数（H × W），即所有需要计算注意力的像素点或局部区域数，token 数量
        #在代码中对应展平后的每个空间位置
        qkv = self.qkv(x)
        #reshape 展平空间维度
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        #flatten 后的空间维度
        #qkv.view(...) 是为了把通道维度按照多头的格式展开，准备做注意力
        #这一操作将 qkv 从 [B, h, H, W] 转成 [B, num_heads, total_dim_per_head, N]，把空间展平成 N，并把通道切分为多头形式
        #拆分 Q, K, V
        #dim=2 是在指定 在哪个维度上进行切分（split）


        #计算注意力分数
        attn = (q.transpose(-2, -1) @ k) * self.scale
        #q.transpose(-2, -1) 把 q 的最后两个维度交换：从 [B, heads, key_dim, N] → [B, heads, N, key_dim]
        #@ k 是矩阵乘法，相当于
        #[B, heads, N, key_dim] @ [B, heads, key_dim, N] → [B, heads, N, N]
        #得到每个 query 位置对所有 key 的相似度
        #缩放系数 self.scale = 1 / sqrt(key_dim) 是为了避免 dot-product 的值太大（防止梯度爆炸），是 Transformer 的标准技巧
        # 第一个 N 是 Query 的位置索引
        #第二个 N 是 Key 的位置索引

        #softmax 得到注意力权重
        attn = attn.softmax(dim=-1)
        #对每个 Query 的 attention scores（对所有 Key 的打分）做 softmax，变成概率分布
        #dim=-1对最后一维进行操作

        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))

        #attn.transpose(-2, -1)：将注意力张量从 [B, heads, N, N] 转换为 [B, heads, N, N]，交换最后两个维度
        #原始是 [query位置, key位置],后来变成[key位置, query位置]
        #为了与 v 对齐
        #v @ attn^T
        #[B, heads, head_dim, N] @ [B, heads, N, N] → [B, heads, head_dim, N]
        #对每个 query 位置，使用注意力权重对所有 value 进行加权求和，得到注意力输出
        #.view(B, C, H, W)：reshape 成 [B, C, H, W] 形式，回到图像结构
        #+ self.pe(v.reshape(B, C, H, W)) 加入位置编码
        #是一个 depthwise 3x3 卷积，用于给每个位置增加空间感知能力（相当于加一点 CNN 的 inductive bias）
        #“注意力结果 + 位置信息” 的融合
        #线性变换输出
        # 是一个 1x1 卷积（或叫 pointwise conv）
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    # 注意力机制 + 前馈网络 + 残差连接
    #用于增强特征提取和注意力机制
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.多头注意力机制
        ffn (nn.Sequential): Feed-forward neural network module.前馈网络（ffn）
        add (bool): Flag indicating whether to add shortcut connections.可选残差连接（shortcut）

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        #c 通道数（输入和输出）
        #attn_ratio Attention中 key 的维度和 head 维度的比例
        #num_heads 多头注意力的头数
        #shortcut 是否使用残差连接（ResNet 风格）
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        #第一个 1x1 卷积扩展通道数（从 c → 2c）
        #第二个 1x1 卷积压缩回 c，实现通道间的信息融合
        #act=False 表示第二层输出没有激活函数
        self.add = shortcut
        #控制是否使用“残差连接（residual add）”

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        #如果 shortcut=True：
        #输出是 x + Attention(x)，即残差连接,否则就直接使用注意力输出
        x = x + self.ffn(x) if self.add else self.ffn(x)
        #再次判断 self.add 使用残差连接：x + FFN(x) 不使用：直接 FFN(x)
        return x



class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc, ec, e=4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor."""
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m) -> None:
        """Initialize residual module with the wrapped module."""
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch, c3, embed):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension."""
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
class ClusterConv(nn.Module):
    def __init__(self, in_channels, out_channels,  bias=False,kernel_size=16):
        super(ClusterConv, self).__init__()
        self.kernels = kernel_size
        self.conv=Convolution(in_channels,in_channels,self.kernels,bias)
    def forward(self, x, edge_index):
        x_j = batched_index_select(x, edge_index)
        x=self.conv(x_j)
        return x

class DyClusterConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1,  bias=False,sub=16):
        super(DyClusterConv2d, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.clustering = GetCluster(kernel_size, dilation)
        self.conv=ClusterConv(in_channels, out_channels, bias,kernel_size=kernel_size)
        self.sub=sub
    def forward(self, x):
        B, C, H, W = x.shape
        # x[b,c,h*W,1]
        # x[b,c,n,1]
        x = x.reshape(B, C, -1, 1).contiguous()
        # Cluster_index [b,n,k*dilaion]
        Cluster_index = self.clustering(x,self.sub)
        # x[b,c,h*w,1]
        x =self.conv(x, Cluster_index)
        return x.reshape(B, H, W, -1).contiguous().permute(0, 3, 1, 2).contiguous()
class DClusConv(nn.Module):

    def __init__(self, in_channels, kernel_size=9, dilation=1,  act=StarReLU,
                 bias=False,sub=16):
        super(DClusConv, self).__init__()
        print(kernel_size,dilation,bias,act,sub)
        self.channels = in_channels
        self.fc1 = nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0,bias=bias)
        self.act=act()
        self.conv = DyClusterConv2d(in_channels*2, in_channels*2 , kernel_size, dilation,
                               bias,sub)
        self.fc2 = nn.Conv2d(in_channels*2, in_channels, 1, stride=1, padding=0,bias=bias)
    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.fc2(x)
        # x = x.permute(0, 2, 3, 1)
        return x
class C3DClus(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.m = nn.Sequential(*[DClusConv(c2) for _ in range(n)])
        self.cv3 = nn.Conv2d(c2 * 2, c2, 1, 1, bias=False)
        self.add = c1 == c2  # 是否可以残差

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), dim=1))
        return x + out if self.add else out
