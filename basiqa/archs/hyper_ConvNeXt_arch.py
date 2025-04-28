import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils.registry import ARCH_REGISTRY


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextBackbone(nn.Module):
    r""" ConvNeXt structure backbone for the hyper network input

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, ffa_out_ch=32, tn_in_ch=384, in_ch=3, feat_size=12,
                 depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        # stem (/4) and downsample(/2)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        # 经过stem层之后从3x384x384变为96x96x96
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # 每次经过一个down192x48x48, 384x24x24, 768x12x12

        # stages (or extracting layers)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # far feature processing layers
        self.fcl_pool = nn.ModuleList()
        self.fcl_fc = nn.ModuleList()
        for ind in range(3):
            sub_pool = nn.Sequential(
                nn.Conv2d(int(dims[ind]), int(dims[ind] / 8), kernel_size=1, stride=1, padding=0, bias=False),
                nn.AvgPool2d(feat_size, stride=feat_size), )
            # 首先进行一次降维，然后进行平均池化例如96x96x96会变为 12x8x8
            # 然后在FC层中，12x8x8变为目标网络所需要的通道数量(32)
            # 第二层第三层的数据分别为 24x4x4 48x2x2 所有的输出都是32通道的
            sub_fc = nn.Linear(int(dims[ind] / 8 * 64 / (4 ** ind)), ffa_out_ch)
            self.fcl_pool.append(sub_pool)
            self.fcl_fc.append(sub_fc)
        # 最后一层无需降维，将剩余特征全部转化为收集为feature
        sub_pool = nn.AvgPool2d(feat_size, stride=feat_size)
        sub_fc = nn.Linear(dims[ind + 1], tn_in_ch - ffa_out_ch * 3)  # 总的通道数对齐到target network input channel
        self.fcl_pool.append(sub_pool)
        self.fcl_fc.append(sub_fc)

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fcl_results = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            fcl_results.append(self.fcl_fc[i](self.fcl_pool[i](x).view(x.size(0), -1)))
        fcl_res = torch.cat((fcl_results[0], fcl_results[1], fcl_results[2], fcl_results[3]), 1)
        # 输出包含两部分，一部分是hyper network要作为参数来源的输入张量(batch*768x12x12)，另一部分则是作为目标网络输入的向量(batch*384)
        out = {}
        out['hyper_in_feat'] = x
        out['target_in_vec'] = fcl_res

        return out

@ARCH_REGISTRY.register()
class HyperNetCN(nn.Module):
    """Hyper network for learning perceptual rules. Based on the convNext backbone with structure modified."""

    def __init__(self, ffa_out_ch=32, tn_in_ch=384, hyper_in_ch=384, in_ch=3, out_channel=1, feat_size=12,
                 depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6, ):
        super(HyperNetCN, self).__init__()
        self.target_input_channel=tn_in_ch
        self.hyper_input_channel=hyper_in_ch
        self.feature_size=feat_size
        self.bb_out_ch=dims[-1]
        self.out_channels=out_channel
        # 基础骨架使用上方的convNext，所有参数可以调节
        self.bbone = ConvNextBackbone(ffa_out_ch, tn_in_ch, in_ch, feat_size, depths, dims, drop_path_rate,
                                      layer_scale_init_value, )

        # 卷积性的头部，将数据降维到应有的输入维度
        # 这一个例子中的tn输入为384，共5层，则每一层的参数量为384x192, 192x96, 96x48, 48x24, 24x1
        # 而第一层中所用的384x192个参数通过384x12x12获得,第二层使用192x12x12获得192x96个参数，以此类推
        # 已经含有激活函数
        self.conv1x1 = nn.ModuleList()
        sub_1x1 = nn.Sequential(nn.Conv2d(dims[-1], hyper_in_ch, stride=1, kernel_size=1, padding=0), nn.GELU(), )
        self.conv1x1.append(sub_1x1)
        for ind in range(4):
            sub_1x1 = nn.Sequential(
                nn.Conv2d(int(hyper_in_ch / 2 ** ind), int(hyper_in_ch / 2 ** (ind + 1)), stride=1, kernel_size=1,
                          padding=0), nn.GELU(), )
            self.conv1x1.append(sub_1x1)

        # 这是得到参数的卷积层和全连接层，分别得到权重和偏置
        # 因为在上一步中已经使用了1x1卷积，所以这一步中的3x3卷积能节省一部分参数，且基本没有信息瓶颈
        #
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.weight_conv = nn.ModuleList()
        self.bias_conv = nn.ModuleList()
        for ind in range(4):
            sub_weight = nn.Conv2d(int(hyper_in_ch / 2 ** ind), int(tn_in_ch ** 2 / 2 / feat_size ** 2 / (4 ** ind)),
                                   stride=1, kernel_size=3, padding=1)
            sub_bias = nn.Linear(int(hyper_in_ch / 2 ** ind), int(tn_in_ch / 2 ** (ind + 1)))
            self.weight_conv.append(sub_weight)
            self.bias_conv.append(sub_bias)
        # 最后一层将输出变为1，最小的1x12x12都明显过量，所以单独添加为一个FC层
        sub_weight = nn.Linear(int(hyper_in_ch / 2 ** 4*feat_size**2), int(tn_in_ch / 2 ** 4)*self.out_channels)
        sub_bias = nn.Linear(int(hyper_in_ch / 2 ** 4*feat_size**2), self.out_channels)
        self.weight_conv.append(sub_weight)
        self.bias_conv.append(sub_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward(self,x):
        bb_out=self.bbone(x)
        tn_in=bb_out['target_in_vec'].view(-1,self.target_input_channel,1,1)
        hyper_in=bb_out['hyper_in_feat'].view(-1,self.bb_out_ch,self.feature_size,self.feature_size)
        # 生成1-4层Fc的参数
        tn_para_w=[]
        tn_para_b=[]
        for ind in range(4):
            hyper_in=self.conv1x1[ind](hyper_in)
            tn_para_w.append(self.weight_conv[ind](hyper_in).view(-1,int(self.target_input_channel/2**(ind+1)),int(self.target_input_channel/2**(ind)),1,1))
            tn_para_b.append(self.bias_conv[ind](self.avgpool(hyper_in).squeeze()).view(-1,int(self.target_input_channel/2**(ind+1))))

        # 最后一层
        hyper_in=torch.flatten(self.conv1x1[4](hyper_in),1)
        tn_para_w.append(self.weight_conv[4](hyper_in).view(-1,self.out_channels,int(self.target_input_channel/2**4),1,1))
        tn_para_b.append(self.bias_conv[4](hyper_in).view(-1,self.out_channels))

        # 规定输出名称
        out = {}
        out['target_in_vec'] = tn_in
        out['target_fc1w'] = tn_para_w[0]
        out['target_fc1b'] = tn_para_b[0]
        out['target_fc2w'] = tn_para_w[1]
        out['target_fc2b'] = tn_para_b[1]
        out['target_fc3w'] = tn_para_w[2]
        out['target_fc3b'] = tn_para_b[2]
        out['target_fc4w'] = tn_para_w[3]
        out['target_fc4b'] = tn_para_b[3]
        out['target_fc5w'] = tn_para_w[4]
        out['target_fc5b'] = tn_para_b[4]

        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
