import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import logging
from .cluster_utils import GetCluster, batched_index_select
from ultralytics.nn.modules.conv import Conv
from .conv import Convolution
from timm.models.metaformer import StarReLU
from .block import Bottleneck
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels,kernels,bias=False):
        super(Convolution, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels,kernels), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(in_channels), requires_grad=True) if bias else None

        trunc_normal_(self.weight, std=.02)
        if bias:
            nn.init.constant_(self.bias, 0)
    def forward(self, x):
        x=(x.transpose(2, 1) * self.weight).sum(dim=-1)
        if self.bias is not None:
            x =x+self.bias

        return x
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

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


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
        # 打印输入通道（实际接收）和内部层通道
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.fc2(x)
        return x


class C3_DClus(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道（确保整数）
        n = int(n)  # 确保块数量为整数

        # 1. 定义 cv1：输入通道 c1 → 输出通道 2*self.c（拆分给两个分支）
        # 注意：DClusConv 此时输入=输出，因此需用 1x1 卷积先调整通道到 2*self.c
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)  # 先用普通卷积调整通道

        # 2. 定义 cv2：输入通道 (2 + n)*self.c（合并后通道）→ 输出通道 c2
        # 由于 DClusConv 输入=输出，需先用它处理后再用 1x1 卷积调整到 c2
        self.cv2 = nn.Sequential(
            DClusConv(in_channels=(2 + n) * self.c),  # DClusConv 保持通道不变
            nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)  # 最终调整到输出通道 c2
        )

        # 3. 定义 Bottleneck 模块列表
        self.m = nn.Sequential(*[
            Bottleneck_DClus(self.c, self.c, shortcut, g)  # 每个 Bottleneck 保持通道 self.c
            for _ in range(n)
        ])

    def forward(self, x):
        # 验证输入通道是否与 cv1 预期一致

        # 拆分特征为两个分支（各 self.c 通道）
        y = list(self.cv1(x).chunk(2, 1))  # 拆分后每个分支通道为 self.c

        # 处理每个 Bottleneck 并拼接
        y.extend(m(y[-1]) for m in self.m)  # 每个 m 输出通道为 self.c

        # 合并所有分支并通过 cv2 输出
        return self.cv2(torch.cat(y, 1))  # 合并后通道为 (2 + n)*self.c

class C3k_DClus(C3_DClus):
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
class Bottleneck_DClus(nn.Module):
    """Standard bottleneck with DClusConv (适配修改后的 DClusConv)"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道（仅用于中间调整）

        # 关键修正：DClusConv 只需要 in_channels（此处为 c1），无需 out_channels
        # 若需要调整通道，在 DClusConv 后加 1x1 卷积
        self.cv1 = nn.Sequential(
            DClusConv(in_channels=c1, kernel_size=k[0]),  # 仅传递 in_channels 和 kernel_size
            nn.Conv2d(c1, c_, 1, groups=g)  # 用 1x1 卷积将通道从 c1 调整为 c_
        )

        self.cv2 = nn.Sequential(
            DClusConv(in_channels=c_, kernel_size=k[1]),  # 输入通道为 c_
            nn.Conv2d(c_, c2, 1, groups=g)  # 最终调整到输出通道 c2
        )

        self.add = shortcut and c1 == c2  # 残差连接要求输入输出通道一致

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DClus(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # 输入分支卷积
        self.cv1 = DClusConv(c1, 2 * self.c)

        # 输出分支卷积
        self.cv2 = DClusConv((2 + n) * self.c, c2)

        # Bottleneck 模块
        self.m = nn.ModuleList(Bottleneck_DClus(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        # 添加通道数验证

        # 分割特征
        y = list(self.cv1(x).chunk(2, 1))

        # 处理每个bottleneck
        for m in self.m:
            y.append(m(y[-1]))

        # 合并并调整通道
        return self.cv2(torch.cat(y, dim=1))


class C3k2_DClus(C2f_DClus):
    """Faster Implementation of CSP Bottleneck with 2 DClusConv convolutions and customizable blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, k=3):
        """
        Initialize C3k2_DClus module with DClusConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks (if True) or Bottleneck_DClus (if False).
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
            k (int): Kernel size for DClusConv operations.
        """
        super().__init__(c1, c2, n, shortcut, g, e)

        # 新增：保存c1和c2作为类属性，供forward方法使用
        self.c1 = c1  # 输入通道
        self.c2 = c2  # 输出通道

        # 根据c3k标志选择使用C3k_DClus或Bottleneck_DClus
        self.m = nn.ModuleList(
            C3k_DClus(self.c, self.c, 2, shortcut, g, e, k) if c3k
            else Bottleneck_DClus(self.c, self.c, shortcut, g, k=(k, k), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        out = super().forward(x)
        return out
