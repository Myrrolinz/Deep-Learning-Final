import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.layers import DropPath
import os
import sys
import math

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
            
# Original LKA
# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)

#         return u * attn

# class LKA(nn.Module):
#     def __init__(self, dim, kernel_size=11):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, kernel_size, stride=1, 
#                                       padding=int(3*(kernel_size-1)/2), groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)
#         return u*attn

# 1. 大核
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 11, stride=1, padding=25, groups=dim, dilation=5)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
    
# 2. 多尺度分支
# class LKA(nn.Module):
#     def __init__(self, dim, kernel_size=5):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 2*kernel_size-1, stride=1, 
#                                       padding=kernel_size * (kernel_size - 1) / 2, groups=dim, dilation=kernel_size)
#         self.conv1 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)
#         return u * attn
    
# class MultiScaleBranch(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)

#     def forward(self, x):
#         out_3x3 = self.conv3x3(x)
#         out_5x5 = self.conv5x5(x)
#         out_7x7 = self.conv7x7(x)
#         return torch.cat([out_3x3, out_5x5, out_7x7], dim=1)


# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
#         self.conv_spatial = MultiScaleBranch(dim)
#         self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = F.relu(attn)
#         attn = self.conv_spatial(attn)
#         attn = F.relu(attn)
#         attn = self.conv1(attn)

#         return u * attn
    
# 3. 增加批归一化
# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.bn0 = nn.BatchNorm2d(dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.bn_spatial = nn.BatchNorm2d(dim)
#         self.conv1 = nn.Conv2d(dim, dim, 1)
#         self.bn1 = nn.BatchNorm2d(dim)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = self.bn0(attn)
#         attn = self.conv_spatial(attn)
#         attn = self.bn_spatial(attn)
#         attn = self.conv1(attn)
#         attn = self.bn1(attn)

#         return u * attn
    
# # 4. 增加非线性激活函数
# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn = F.gelu(attn)
#         attn = self.conv_spatial(attn)
#         attn = F.gelu(attn)
#         attn = self.conv1(attn)

#         return u * attn

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


model_urls = {
    "van_b0": "https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar",
    "van_b1": "https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar",
    "van_b2": "https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar",
    "van_b3": "https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar",
}


def load_model_weights(model, arch, kwargs):
    url = model_urls[arch]
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )
    strict = True
    if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model


@register_model
def van_b0(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b0", kwargs)
    return model


@register_model
def van_b1(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b1", kwargs)
    return model


@register_model
def van_b2(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b2", kwargs)
    return model


@register_model
def van_b3(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b3", kwargs)
    return model


@register_model
def van_b4(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b4", kwargs)
    return model


@register_model
def van_b5(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b5", kwargs)
    return model


@register_model
def van_b6(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6, 6, 90, 6],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b6", kwargs)
    return model
